"""tools/video_flat_stats.py: the flat-wall definitions applied to the per-tick floor/ceiling shares recorded while capturing the video."""
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


def test_video_stats_count_flat_ticks_and_ticks_stuck_for_32_in_a_row():
    import video_flat_stats as V
    n = 100
    dream = [0.3] * n
    real = [0.3] * n
    for t in range(40, 100):
        dream[t] = 0.0                       # the dream sits on a flat wall from tick 40 on
    for t in range(10, 20):
        real[t] = 0.0                        # the real game faces a wall for 10 ticks
    s = V.stats({"bg_dream_per_tick": dream, "bg_real_per_tick": real, "wake_every": 0})
    assert s["ticks"] == 100 and s["dream_flat_share"] == pytest.approx(0.6) and s["real_flat_share"] == pytest.approx(0.1)
    assert s["dream_stuck_share"] == pytest.approx(29 / 100)                            # flat for the last 32 ticks from tick 71 on
    assert s["real_stuck_share"] == 0.0 and s["dream_stuck_ticks_while_real_is_not"] == 29
