"""eval_prereg.py: the evaluation protocol of docs/PREREG-v3.md (common wake-up frame, per-episode tracking horizon,
collapse rate, spin-return), plus the offline `analyze` that recomputes everything that does not need a GPU."""
import functools
import json
import pathlib
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_analyze_reports_tracking_horizon_and_collapse_from_per_frame_arrays():
    import eval_prereg as P
    E, L = 4, 520
    psnr = np.full((E, L), 30.0)
    psnr[1:, 20:] = 12.0                                    # episodes 1-3 lose the real game at frame 21
    bg_real = np.full((E, L), 0.4)                          # the real game never faces a wall
    bg_dream = np.full((E, L), 0.4)
    bg_dream[2, 100:] = 0.0                                 # episode 2 collapses onto a flat wall from frame 101 to the end
    a = P.analyze(psnr, bg_dream, bg_real, window=16)
    assert a["tracking_horizon_20db"]["median"] == 19 and a["tracking_horizon_20db"]["share_never_below"] == 0.25
    # frames 101..520 flat; stuck from frame 116; counted frames: t in 64..512 -> 116..512 = 397 of 449 for that one episode
    assert a["collapse"]["rate"] == pytest.approx(397 / 449 / 4)
    assert a["collapse"]["window"] == 16 and len(a["collapse"]["rate_ci95"]) == 2
    assert a["collapse_plain"]["rate"] == pytest.approx(412 / 449 / 4)              # every flat frame counts when the window is 1
    assert a["real_flat_share"] == 0.0 and a["dream_flat_share"] == pytest.approx(412 / 449 / 4)       # shares over frames 64..512


def test_analyze_keeps_the_per_episode_horizons_so_two_models_can_be_compared_episode_by_episode():
    import eval_prereg as P
    L = 60
    good = np.full((3, L), 30.0)
    bad = good.copy()
    bad[0, 10:] = 10.0
    bg = np.zeros((3, L)) + 0.5
    a, b = P.analyze(good, bg, bg), P.analyze(bad, bg, bg)
    assert a["tracking_horizon_20db"]["per_episode"] == [L + 1] * 3
    assert b["tracking_horizon_20db"]["per_episode"][1:] == [L + 1] * 2 and b["tracking_horizon_20db"]["per_episode"][0] < L


def test_eval_prereg_runs_end_to_end_on_a_tiny_8_context_checkpoint(tmp_path, monkeypatch):
    import eval_prereg as P
    ck_dir = tmp_path / "run"
    r = subprocess.run([sys.executable, str(ROOT / "train.py"), "--out", str(ck_dir), "--device", "cpu", "--channels", "16,16,16,16",
                        "--depths", "1,1,1,1", "--cond", "64", "--batch", "4", "--workers", "0", "--eval-every", "0",
                        "--steps", "2", "--num-ctx", "8"], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    (tmp_path / "results").mkdir()
    monkeypatch.setattr(P, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["eval_prereg.py", "--ckpt", str(ck_dir / "ckpt_last.pt"), "--name", "t8", "--device", "cpu",
                                      "--episodes", "2", "--n-traj", "4", "--horizon", "70", "--spin-n", "2", "--n-val", "4", "--keep", "1"])
    P.main()
    out = json.loads((tmp_path / "results" / "prereg_t8.json").read_text())
    assert out["num_ctx"] == 8 and out["protocol"]["wake_frame"] == 8 and out["protocol"]["episodes"] == 2
    assert out["tracking_horizon_20db"]["episodes"] == 2 and "rate" in out["collapse"]
    assert np.isfinite(out["spin_360"]["dream_frame72_vs_start_view_psnr"]) and np.isfinite(out["one_step"]["psnr"])
    # the reference points that make PSNR levels readable across games: freezing the last real frame, and an unrelated real frame
    assert set(out["baseline_freeze_last_real_frame"]["psnr_at"]) >= {"1", "8", "32", "64"}
    assert 5 < out["baseline_unrelated_frame_psnr"] < 25
    z = np.load(tmp_path / "results" / "prereg_t8.npz")
    assert z["psnr"].shape == (2, 70) and z["bg_dream"].shape == (2, 70) and z["bg_real"].shape == (2, 70)
    # offline re-analysis from the saved arrays reproduces the same numbers
    again = P.analyze(z["psnr"], z["bg_dream"], z["bg_real"], window=out["collapse"]["window"])
    assert again["collapse"]["rate"] == out["collapse"]["rate"]
    assert again["tracking_horizon_20db"]["median"] == out["tracking_horizon_20db"]["median"]
