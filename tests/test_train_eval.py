"""Tests for the evaluation helpers (model/evaluate.py) and the resumable training script (train.py)."""
import json
import subprocess
import sys
import pathlib

import numpy as np
import pytest
import torch

from game import raycaster as rc
from model import evaluate as ev
from model.denoiser import Denoiser, DenoiserConfig, encode_actions

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def world():
    return rc.World()


def test_val_windows_are_fixed_and_shaped(world):
    a = ev.val_windows(world, n=6, seed=5)
    b = ev.val_windows(world, n=6, seed=5)
    assert a["frames"].shape == (6, 5, 3, 64, 64) and a["frames"].dtype == torch.float32
    assert a["actions"].shape == (6, 4, 3)
    assert torch.equal(a["frames"], b["frames"])
    assert a["frames"].min() >= -1 and a["frames"].max() <= 1


def test_one_step_metrics_of_a_perfect_denoiser_are_near_perfect(world):
    w = ev.val_windows(world, n=4, seed=1)
    gt = w["frames"][:, 4]

    def oracle(x, sigma, ctx, act, s_c):
        return gt

    m = ev.one_step_metrics(oracle, w, steps=3, order=1, ctx_sigma=0.0)
    assert m["psnr"] > 60 and m["ssim"] > 0.999


def test_one_step_metrics_penalise_copying_the_last_frame_less_than_noise(world):
    w = ev.val_windows(world, n=8, seed=2)

    def copy_last(x, sigma, ctx, act, s_c):
        return ctx[:, -1]

    def noise(x, sigma, ctx, act, s_c):
        return torch.rand_like(x) * 2 - 1

    a = ev.one_step_metrics(copy_last, w, steps=1, order=1, ctx_sigma=0.0)
    b = ev.one_step_metrics(noise, w, steps=1, order=1, ctx_sigma=0.0)
    assert a["psnr"] > b["psnr"] + 5


def test_rollout_metrics_have_one_value_per_generated_frame(world):
    t = ev.eval_trajectories_t(world, n=3, length=14, seed=3)
    def copy_last(x, sigma, ctx, act, s_c):
        return ctx[:, -1]
    r = ev.rollout_metrics(copy_last, t, horizon=10, steps=1, order=1, ctx_sigma=0.0)
    assert r["psnr"].shape == (10,) and r["ssim"].shape == (10,)
    assert np.isfinite(r["psnr"]).all()
    # copying the last context frame forever must get worse (on average) as the true camera moves on
    assert r["psnr"][0] > r["psnr"][-1]


def run_train(tmp, *extra):
    cmd = [sys.executable, str(ROOT / "train.py"), "--out", str(tmp), "--device", "cpu", "--channels", "16,16,16,16",
           "--depths", "1,1,1,1", "--cond", "64", "--batch", "4", "--workers", "0", "--eval-every", "0",
           "--log-every", "2", "--ckpt-every", "3", *extra]
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=300)


def test_train_script_writes_checkpoints_logs_and_resumes(tmp_path):
    r = run_train(tmp_path, "--steps", "6")
    assert r.returncode == 0, r.stderr[-2000:]
    ck = torch.load(tmp_path / "ckpt_last.pt", map_location="cpu", weights_only=False)
    assert ck["step"] == 6 and "ema" in ck and "model" in ck and "opt" in ck and "config" in ck
    rows = [json.loads(l) for l in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert rows and all(np.isfinite(r_["loss"]) for r_ in rows)
    # resume and continue to step 10
    r = run_train(tmp_path, "--steps", "10", "--resume")
    assert r.returncode == 0, r.stderr[-2000:]
    ck2 = torch.load(tmp_path / "ckpt_last.pt", map_location="cpu", weights_only=False)
    assert ck2["step"] == 10
    steps = [json.loads(l)["step"] for l in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert steps == sorted(steps) and steps[-1] == 10


def test_train_script_stops_on_time_limit_and_still_checkpoints(tmp_path):
    r = run_train(tmp_path, "--steps", "100000", "--time-limit-min", "0.15")
    assert r.returncode == 0, r.stderr[-2000:]
    ck = torch.load(tmp_path / "ckpt_last.pt", map_location="cpu", weights_only=False)
    assert 0 < ck["step"] < 100000


def test_train_script_supports_unrolled_training(tmp_path):
    r = run_train(tmp_path, "--steps", "4", "--unroll", "3")
    assert r.returncode == 0, r.stderr[-2000:]
    ck = torch.load(tmp_path / "ckpt_last.pt", map_location="cpu", weights_only=False)
    assert ck["step"] == 4
    rows = [json.loads(l) for l in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert all(np.isfinite(r_["loss"]) for r_ in rows)


# ------------------------------------------------------------------ "is the dream still a valid game frame?" metric

def test_nearest_real_frame_psnr_ranks_real_noisy_and_random_frames(world):
    bank = ev.make_frame_bank(world, n=600, seed=1)
    t = ev.eval_trajectories_t(world, n=2, length=8, seed=2)
    real_in_bank = bank["frames"][:5]
    p_bank = ev.nearest_real_psnr(real_in_bank, bank)
    assert (p_bank >= 45).all(), "a frame that is in the bank has an exact match (GEMM round-off caps the value near 50-60 dB)"
    noisy = (real_in_bank + 0.2 * torch.randn_like(real_in_bank)).clamp(-1, 1)
    p_noisy = ev.nearest_real_psnr(noisy, bank)
    p_rand = ev.nearest_real_psnr(torch.rand(5, 3, 64, 64) * 2 - 1, bank)
    assert (p_noisy > p_rand + 3).all() and (p_noisy < p_bank).all()
    held_out = ev.nearest_real_psnr(t["frames"][:, 4], bank)         # real frames not in the bank
    assert held_out.mean() > p_rand.mean() + 3


def test_train_script_can_start_from_another_runs_ema_weights(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    assert run_train(a, "--steps", "4").returncode == 0
    r = run_train(b, "--steps", "3", "--init-from", str(a / "ckpt_last.pt"))
    assert r.returncode == 0, r.stderr[-2000:]
    ck = torch.load(b / "ckpt_last.pt", map_location="cpu", weights_only=False)
    src = torch.load(a / "ckpt_last.pt", map_location="cpu", weights_only=False)
    assert ck["step"] == 3                                    # a fresh step counter and optimiser ...
    k = next(k for k in src["ema"] if "conv_in.weight" in k)
    assert (ck["model"][k] - src["ema"][k]).abs().max() < 0.05   # ... but the weights started from the source EMA


def test_train_script_supports_rollout_context_training(tmp_path):
    r = run_train(tmp_path, "--steps", "4", "--rollout-ctx", "2", "--rollout-prob", "0.7")
    assert r.returncode == 0, r.stderr[-2000:]
    ck = torch.load(tmp_path / "ckpt_last.pt", map_location="cpu", weights_only=False)
    assert ck["step"] == 4


# ------------------------------------------------------------------ eval.py helpers (spin test, tracking horizon, sharpness)

def test_first_below_and_grad_mag_helpers():
    import eval as E
    assert E.first_below([30, 28, 24, 19, 18], 25.0) == 3
    assert E.first_below([30, 28], 25.0) == 3                      # never below: len + 1
    flat = torch.zeros(2, 3, 8, 8)
    stripes = flat.clone()
    stripes[..., ::2] = 1.0
    assert E.grad_mag(flat) == 0 and E.grad_mag(stripes) > 0.4


def test_spin_test_with_a_frozen_dream_never_returns_a_different_view(world):
    """A denoiser that always repeats the last context frame stays on the start view: it matches the start view exactly
    at frame 72 (the real game also returns to it after 72 x 5 degrees), but not the real frames in between."""
    import eval as E

    def copy_last(x, sigma, ctx, act, s_c):
        return ctx[:, -1]

    r = E.spin_test(world, copy_last, n=3, steps=1, ctx_sigma=0.0)
    assert r["real_frame72_vs_start_view_psnr"] >= 99, "the real game closes the loop after 72 turns of 5 degrees"
    assert r["dream_frame72_vs_start_view_psnr"] >= 99
    assert r["dream_psnr_at"]["24"] < 25


def test_hold_one_key_rollouts_report_every_control(world):
    import eval as E

    def copy_last(x, sigma, ctx, act, s_c):
        return ctx[:, -1]

    r = E.hold_action_rollouts(world, copy_last, n=3, horizon=8, steps=1, ctx_sigma=0.0)
    assert set(r) == set(E.HOLD_ACTIONS)
    for k, v in r.items():
        assert set(v["dream_psnr_at"]) >= {"1", "4", "8"} and np.isfinite(list(v["freeze_psnr_at"].values())).all()
        # copying the last frame IS the freeze baseline
        assert abs(v["dream_psnr_at"]["8"] - v["freeze_psnr_at"]["8"]) < 1e-3
