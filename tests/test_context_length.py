"""Context length K (v3 ablation): the model, the evaluation helpers, training and export take K from the config.
K = 4 is v2 and stays the default; every test here that uses K = 4 also pins that nothing about v2 changed."""
import functools
import json
import pathlib
import subprocess
import sys

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

import export as ex
from game import raycaster as rc
from model import evaluate as ev
from model.denoiser import Denoiser, DenoiserConfig, encode_actions

ROOT = pathlib.Path(__file__).resolve().parents[1]
TINY = dict(channels=(16, 16, 16, 16), depths=(1, 1, 1, 1), cond_channels=64)


@pytest.fixture(scope="module")
def world():
    return rc.World()


def copy_last(x, sigma, ctx, act, s_c):
    return ctx[:, -1]


# ------------------------------------------------------------------ model

def test_denoiser_with_8_context_frames_has_a_wider_input_conv_and_trains():
    m = Denoiser(DenoiserConfig(**TINY, num_ctx=8))
    assert m.inner.conv_in.in_channels == 27
    g = torch.Generator().manual_seed(0)
    ctx = torch.rand(2, 8, 3, 64, 64, generator=g) * 2 - 1
    tgt = torch.rand(2, 3, 64, 64, generator=g) * 2 - 1
    act = encode_actions(torch.randint(0, 3, (2, 8, 3), generator=g))
    loss, _ = m.loss(ctx, tgt, act)
    assert torch.isfinite(loss)
    out = m.denoise_flat(tgt, torch.full((2,), 0.7), ctx.reshape(2, 24, 64, 64), act, torch.zeros(2))
    assert out.shape == (2, 3, 64, 64)


# ------------------------------------------------------------------ evaluation windows and one-step metrics

def test_val_windows_take_the_context_length(world):
    w4 = ev.val_windows(world, n=5, seed=7)
    w8 = ev.val_windows(world, n=5, seed=7, K=8)
    assert w4["frames"].shape == (5, 5, 3, 64, 64) and w4["actions"].shape == (5, 4, 3)
    assert w8["frames"].shape == (5, 9, 3, 64, 64) and w8["actions"].shape == (5, 8, 3)
    # same seed, same games: the first frames agree, so K = 4 and K = 8 models can be compared on the same states
    assert torch.equal(w8["frames"][:, :5], w4["frames"])


def test_one_step_metrics_use_the_last_K_frames_before_the_target(world):
    w = ev.val_windows(world, n=4, seed=3, K=8)
    seen = {}

    def spy(x, sigma, ctx, act, s_c):
        seen["ctx"], seen["act"] = ctx.clone(), act.clone()
        return ctx[:, -1]

    for K in (4, 8):
        ev.one_step_metrics(spy, w, steps=1, ctx_sigma=0.0, K=K)
        assert seen["ctx"].shape == (4, K, 3, 64, 64) and seen["act"].shape == (4, K, 9)
        assert torch.equal(seen["ctx"], w["frames"][:, 8 - K:8])          # context ends right before the target (frame 8)
        assert torch.equal(seen["act"], encode_actions(w["actions"][:, 8 - K:8]))
    # copy-last on the same target frame is the same baseline for every K
    a = ev.one_step_metrics(copy_last, w, steps=1, ctx_sigma=0.0, K=4)
    b = ev.one_step_metrics(copy_last, w, steps=1, ctx_sigma=0.0, K=8)
    assert abs(a["psnr"] - b["psnr"]) < 1e-6


def test_one_step_metrics_default_K_is_the_window_length_minus_one(world):
    """Backwards compatibility: 5-frame windows are K = 4 (v2) without saying so."""
    w = ev.val_windows(world, n=4, seed=3)
    a = ev.one_step_metrics(copy_last, w, steps=1, ctx_sigma=0.0)
    b = ev.one_step_metrics(copy_last, w, steps=1, ctx_sigma=0.0, K=4)
    assert a["psnr"] == b["psnr"]


# ------------------------------------------------------------------ rollouts from a common wake-up frame

def test_rollout_from_slices_every_model_to_the_same_first_dreamed_frame(world):
    """All models wake up at the same real frame, whatever their K: a K = 8 model sees 8 real frames before it, a K = 4
    model the last 4 of those, and both are scored against the same real frames."""
    from model.evaluate import wake_slice
    t = ev.eval_trajectories_t(world, n=3, length=8 + 10, seed=3)
    for K in (4, 8):
        s = wake_slice(t, wake=8, K=K)
        assert s["frames"].shape[1] == K + 10 and s["actions"].shape[1] == K + 10 - 1
        assert torch.equal(s["frames"][:, K], t["frames"][:, 8])                 # first dreamed frame = real frame 8
        r = ev.rollout_metrics(copy_last, s, horizon=10, steps=1, ctx_sigma=0.0, K=K)
        assert r["psnr"].shape == (10,)
    assert not torch.equal(wake_slice(t, 8, 4)["frames"][:, 0], wake_slice(t, 8, 8)["frames"][:, 0])


# ------------------------------------------------------------------ eval.py probes that build their own context

def test_spin_test_and_hold_keys_use_K_standing_frames_as_context(world):
    import eval as E
    r = E.spin_test(world, copy_last, n=3, steps=1, ctx_sigma=0.0, K=8)
    assert r["real_frame72_vs_start_view_psnr"] >= 99 and r["dream_frame72_vs_start_view_psnr"] >= 99
    h = E.hold_action_rollouts(world, copy_last, n=3, horizon=8, steps=1, ctx_sigma=0.0, K=8)
    for v in h.values():
        assert abs(v["dream_psnr_at"]["8"] - v["freeze_psnr_at"]["8"]) < 1e-3


# ------------------------------------------------------------------ training

def run_train(tmp, *extra):
    cmd = [sys.executable, str(ROOT / "train.py"), "--out", str(tmp), "--device", "cpu", "--channels", "16,16,16,16",
           "--depths", "1,1,1,1", "--cond", "64", "--batch", "4", "--workers", "0", "--eval-every", "0",
           "--log-every", "2", "--ckpt-every", "3", *extra]
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=300)


@pytest.mark.parametrize("extra", [(), ("--unroll", "2"), ("--rollout-ctx", "2")])
def test_train_script_supports_8_context_frames(tmp_path, extra):
    r = run_train(tmp_path, "--steps", "4", "--num-ctx", "8", *extra)
    assert r.returncode == 0, r.stderr[-2000:]
    ck = torch.load(tmp_path / "ckpt_last.pt", map_location="cpu", weights_only=False)
    assert ck["step"] == 4 and ck["config"]["num_ctx"] == 8
    assert ck["ema"]["inner.conv_in.weight"].shape[1] == 27
    rows = [json.loads(l) for l in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert rows and all(np.isfinite(x["loss"]) for x in rows)


def test_train_script_evaluates_8_context_models_on_8_context_windows(tmp_path):
    r = run_train(tmp_path, "--steps", "2", "--num-ctx", "8", "--eval-every", "2", "--eval-n", "4")
    assert r.returncode == 0, r.stderr[-2000:]
    assert "eval (EMA" in r.stdout


def test_default_context_length_is_still_4(tmp_path):
    r = run_train(tmp_path, "--steps", "2")
    assert r.returncode == 0, r.stderr[-2000:]
    ck = torch.load(tmp_path / "ckpt_last.pt", map_location="cpu", weights_only=False)
    assert ck["config"]["num_ctx"] == 4 and ck["ema"]["inner.conv_in.weight"].shape[1] == 15


def test_train_script_records_and_uses_the_game_spec(tmp_path):
    """--game picks the game (spec json) the data is generated from; the checkpoint remembers it so that evaluation
    and export use the same game."""
    spec = ROOT / "game" / "spec.json"
    r = run_train(tmp_path, "--steps", "2", "--game", str(spec))
    assert r.returncode == 0, r.stderr[-2000:]
    ck = torch.load(tmp_path / "ckpt_last.pt", map_location="cpu", weights_only=False)
    assert pathlib.Path(ck["args"]["game"]).name == "spec.json"
    # the final EMA checkpoint is what evaluation and export load: it must carry the game too (a v3 model evaluated on the v2 game
    # would be silently wrong)
    assert ex.ckpt_game(str(tmp_path / "final_ema.pt")).endswith("spec.json")


# ------------------------------------------------------------------ export

def test_export_with_8_context_frames_has_the_wider_static_interface_and_matches_pytorch(tmp_path):
    torch.manual_seed(0)
    m = Denoiser(DenoiserConfig(channels=(16, 32, 48, 64), depths=(1, 1, 1, 1), cond_channels=64, num_ctx=8)).eval()
    torch.nn.init.normal_(m.inner.conv_out.weight, std=0.05)
    p = tmp_path / "k8.onnx"
    ex.export_onnx(m, str(p))
    g = onnx.load(str(p))
    ins = {i.name: [d.dim_value for d in i.type.tensor_type.shape.dim] for i in g.graph.input}
    assert ins == {"x": [1, 3, 64, 64], "sigma": [1], "ctx": [1, 24, 64, 64], "act": [1, 8, 9], "ctx_sigma": [1]}
    xs = ex.example_inputs(0, 0.7, 0.1, K=8)
    sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    out = sess.run(None, {n: v.numpy().astype(np.float32) for n, v in zip(ex.INPUTS, xs)})[0]
    with torch.no_grad():
        ref = m.denoise_flat(*xs).numpy()
    assert np.abs(out - ref).max() < 2e-4


def test_real_export_cases_follow_the_models_context_length():
    torch.manual_seed(0)
    m = Denoiser(DenoiserConfig(**TINY, num_ctx=8)).eval()
    cases = ex.real_cases(m, n=2)
    x, sigma, ctx, act, cs = cases[0]
    assert ctx.shape == (1, 24, 64, 64) and act.shape == (1, 8, 9)


# ------------------------------------------------------------------ eval.py end to end on a tiny 8-context checkpoint

def test_eval_script_runs_end_to_end_for_an_8_context_checkpoint(tmp_path, monkeypatch):
    import eval as E
    ck_dir = tmp_path / "run"
    r = run_train(ck_dir, "--steps", "2", "--num-ctx", "8", "--game", "game/spec_v2.json")
    assert r.returncode == 0, r.stderr[-2000:]
    (tmp_path / "results").mkdir()
    monkeypatch.setattr(E, "ROOT", tmp_path)
    monkeypatch.setattr(E, "spin_test", functools.partial(E.spin_test, n=2))               # the probes have their own K test above;
    monkeypatch.setattr(E, "hold_action_rollouts", functools.partial(E.hold_action_rollouts, n=2, horizon=8))   # keep this one fast
    monkeypatch.setattr(sys, "argv", ["eval.py", "--ckpt", str(ck_dir / "ckpt_last.pt"), "--name", "t8", "--device", "cpu",
                                      "--no-figs", "--quick", "--episodes", "2", "--long", "12", "--n-val", "4", "--bank", "100",
                                      "--steps", "1", "--ctx-sigma", "0.05"])
    E.main()
    out = json.loads((tmp_path / "results" / "eval_t8.json").read_text())
    assert out["num_ctx"] == 8 and out["game"].endswith("spec_v2.json")
    assert len(out["rollout_primary"]["psnr_curve"]) == 12
    assert out["spin_360"]["episodes"] == 2 and np.isfinite(out["spin_360"]["dream_frame72_vs_start_view_psnr"])


def test_a_final_checkpoint_without_args_takes_the_game_from_the_resume_checkpoint_next_to_it(tmp_path):
    """Runs started before final_ema.pt carried its args: their game is in ckpt_last.pt in the same folder."""
    cfg = {"channels": [16, 16, 16, 16]}
    torch.save({"ema": {}, "config": cfg, "step": 5}, tmp_path / "final_ema.pt")
    assert ex.ckpt_game(str(tmp_path / "final_ema.pt")) == "game/spec_v2.json"          # nothing recorded anywhere: the v2 game
    torch.save({"ema": {}, "config": cfg, "step": 5, "args": {"game": "game/spec_v3.json"}}, tmp_path / "ckpt_last.pt")
    assert ex.ckpt_game(str(tmp_path / "final_ema.pt")) == "game/spec_v3.json"


def test_export_cli_can_write_a_random_8_context_model_for_browser_tests(tmp_path):
    """`export.py --random-init --num-ctx 8`: a model of the right interface without training, to test the browser pipeline for K = 8."""
    report = ROOT / "results" / "onnx_export_dream.json"                    # the real model's export report must not be touched
    before = report.read_text() if report.exists() else None
    r = subprocess.run([sys.executable, str(ROOT / "export.py"), "--random-init", "--num-ctx", "8", "--channels", "16,16,16,16", "--depths", "1,1,1,1",
                        "--out", str(tmp_path), "--name", "dream"], cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    assert (report.read_text() if report.exists() else None) == before
    assert (tmp_path / "onnx_export_dream.json").exists()
    meta = json.loads((tmp_path / "dream_meta.json").read_text())
    assert meta["config"]["num_ctx"] == 8
    g = onnx.load(str(tmp_path / "dream_fp32.onnx"))
    ins = {i.name: [d.dim_value for d in i.type.tensor_type.shape.dim] for i in g.graph.input}
    assert ins["ctx"] == [1, 24, 64, 64] and ins["act"] == [1, 8, 9]


# ------------------------------------------------------------------ review findings: reproducibility and guards

def test_one_step_metrics_are_reproducible_for_a_seed_and_differ_between_seeds(world):
    """The seed argument used to be ignored (the global torch RNG drove the context noise and the sampler): two identical calls gave
    different PSNRs."""
    torch.manual_seed(0)
    m = Denoiser(DenoiserConfig(**TINY, num_ctx=4)).eval()
    torch.nn.init.normal_(m.inner.conv_out.weight, std=0.05)
    w = ev.val_windows(world, n=6, seed=2)
    a = ev.one_step_metrics(m.denoise, w, steps=1, ctx_sigma=0.1, init="unit", seed=3)
    torch.manual_seed(123)                                  # a different global state must not matter
    b = ev.one_step_metrics(m.denoise, w, steps=1, ctx_sigma=0.1, init="unit", seed=3)
    c = ev.one_step_metrics(m.denoise, w, steps=1, ctx_sigma=0.1, init="unit", seed=4)
    assert a["psnr"] == b["psnr"] and a["psnr"] != c["psnr"]


def test_resuming_a_run_on_a_different_game_is_refused(tmp_path):
    r = run_train(tmp_path, "--steps", "2", "--game", "game/spec_v2.json")
    assert r.returncode == 0, r.stderr[-2000:]
    r = run_train(tmp_path, "--steps", "4", "--resume", "--game", "game/spec_v3.json")
    assert r.returncode != 0 and "game" in (r.stderr + r.stdout).lower()


def test_eval_prereg_refuses_a_game_that_contradicts_the_checkpoint(tmp_path, monkeypatch):
    import eval_prereg as P
    run = tmp_path / "run"
    r = run_train(run, "--steps", "2", "--game", "game/spec_v3.json")
    assert r.returncode == 0, r.stderr[-2000:]
    monkeypatch.setattr(P, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["eval_prereg.py", "--ckpt", str(run / "final_ema.pt"), "--name", "x", "--game", "game/spec_v2.json",
                                      "--device", "cpu", "--episodes", "2", "--n-traj", "2", "--horizon", "10"])
    with pytest.raises(SystemExit) as e:
        P.main()
    assert "game" in str(e.value).lower()
