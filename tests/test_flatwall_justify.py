"""tools/flatwall_justify.py: the table that justifies the flat-wall constants of docs/PREREG-v3.md from v2 rollouts."""
import json
import pathlib
import subprocess
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


def synthetic(tmp_path):
    E, L = 6, 520
    bg_real = np.full((E, L), 0.3, dtype=np.float32)
    bg_dream = np.full((E, L), 0.3, dtype=np.float32)
    bg_real[:, 200:210] = 0.0                       # a short flat spell in the real game: 10 frames
    bg_dream[:3, 100:] = 0.0                        # half of the dream episodes collapse for good from frame 101
    np.savez(tmp_path / "p.npz", psnr=np.full((E, L), 20.0, dtype=np.float32), bg_dream=bg_dream, bg_real=bg_real)
    return tmp_path / "p.npz"


def test_justification_tables_describe_flat_shares_runs_and_windows(tmp_path):
    import flatwall_justify as J
    out = J.justify(np.load(synthetic(tmp_path)))
    assert out["flat_share_by_max_bg"]["0.05"]["dream"] == pytest.approx(3 * 412 / (6 * 449))     # frames 101..512 flat in 3 episodes
    assert out["flat_share_by_max_bg"]["0.05"]["real"] == pytest.approx(10 / 449)
    rows = {r["window"]: r for r in out["window_table"]}
    assert rows[1]["real_stuck_share"] == pytest.approx(10 / 449)
    assert rows[32]["real_stuck_share"] == 0.0 and rows[32]["dream_stuck_share"] > 0.4          # 10-frame real spells never count at W = 32
    assert out["run_lengths"]["real"]["max"] == 10 and out["run_lengths"]["dream"]["max"] == 420
    assert out["flat_share_by_time"]["1-32"]["dream"] == 0.0 and out["flat_share_by_time"]["257-512"]["dream"] == pytest.approx(0.5)
    assert out["chosen"]["window"] == 32 and out["chosen"]["max_bg"] == 0.05 and out["chosen"]["bg_tol"] == 4


def test_tolerance_table_uses_the_saved_frames_when_given(tmp_path):
    import flatwall_justify as J
    from game import raycaster as rc
    w = rc.World()
    st = rc.state_from_pose(np.array([8.5, 8.5]), np.array([5.5, 9.5]), np.radians(np.array([-90.0, -90.0])))
    fr = w.render(st)                                                  # a wall close-up and a corridor-ish view
    rng = np.random.default_rng(0)
    noisy = np.clip(fr.astype(int) + rng.integers(-1, 2, fr.shape), 0, 255).astype(np.uint8)
    t = J.tolerance_table(noisy[None], w.bg_rows, tols=(0, 2, 4, 10))
    assert t["4"] == t["10"] == t["2"]                       # model error of 1 level is absorbed from tolerance 2 on ...
    assert t["0"]["flat"] > t["2"]["flat"]                  # ... while an exact match misses the noisy floor and wrongly calls the view flat
