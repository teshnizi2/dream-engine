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
