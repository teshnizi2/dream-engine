"""tools/flat_spells.py: how long do flat-wall spells last in a dream and in the real game (exploratory, from the saved per-frame arrays)?"""
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


def test_spell_statistics_and_the_chance_of_leaving_a_flat_wall():
    import flat_spells as F
    bg = np.full((2, 40), 0.3)
    bg[0, 5:15] = 0.0                       # a 10-frame spell
    bg[1, 0:40] = 0.0                       # a spell that never ends
    s = F.spells(bg, t_from=1)
    assert s["n_spells"] == 2 and s["longest"] == 40 and s["share_of_frames_flat"] == pytest.approx(50 / 80)
    assert s["n_ended"] == 1 and s["median_length_of_ended"] == 10
    # of the 50 flat frames, 49 have a successor in the rollout (the last frame of an episode has none); exactly one of those is followed by an open frame
    assert s["chance_of_leaving_per_flat_frame"] == pytest.approx(1 / 49)


def test_only_frames_from_the_window_start_on_are_counted():
    import flat_spells as F
    bg = np.full((1, 100), 0.3)
    bg[0, :50] = 0.0
    s = F.spells(bg, t_from=64)             # frames 64..100: all open
    assert s["n_spells"] == 0 and s["share_of_frames_flat"] == 0.0
