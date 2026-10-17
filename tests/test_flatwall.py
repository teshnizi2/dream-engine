"""The flat-wall metric of docs/PREREG-v3.md (model/flatwall.py): which frames are 'flat wall' (no floor/ceiling visible),
when a dream is 'stuck' on one, and the collapse rate against the matching real frames."""
import numpy as np
import pytest

from game import raycaster as rc
from model import flatwall as fw


@pytest.fixture(scope="module")
def world():
    return rc.World()


def S(x, y, deg=0.0):
    return rc.state_from_pose(np.array([x]), np.array([y]), np.radians(np.array([deg])))


# ---------------------------------------------------------------- per-frame background fraction

def test_wall_close_up_has_no_background_and_a_corridor_view_has_plenty(world):
    close = world.render(S(8.5, 5.5, -90.0))[0]           # 0.5 cells from the hall's north wall, facing it: the wall fills the view
    far = world.render(S(8.5, 9.5, -90.0))[0]             # same wall from 4.5 cells: a 14-row band between ceiling and floor
    assert fw.background_fraction(close, world.bg_rows) == 0.0
    assert fw.background_fraction(far, world.bg_rows) > 0.6


def test_background_fraction_ignores_a_few_grey_levels_of_model_error(world):
    """Dreamed frames reproduce the floor and ceiling to within a level or two: that must not change the verdict."""
    rng = np.random.default_rng(0)
    f = world.render(S(8.5, 9.5, -90.0))[0]
    noisy = np.clip(f.astype(int) + rng.integers(-2, 3, f.shape), 0, 255).astype(np.uint8)
    assert fw.background_fraction(noisy, world.bg_rows) == fw.background_fraction(f, world.bg_rows)
    very_noisy = np.clip(f.astype(int) + rng.integers(-20, 21, f.shape), 0, 255).astype(np.uint8)
    assert fw.background_fraction(very_noisy, world.bg_rows) < 0.5 * fw.background_fraction(f, world.bg_rows)


def test_background_fraction_works_on_batches_of_frames(world):
    fr = np.stack([world.render(S(8.5, 5.5, -90.0))[0], world.render(S(8.5, 9.5, -90.0))[0]])
    fr = np.stack([fr, fr[::-1]])                                   # [2 episodes, 2 frames, H, W, 3]
    bg = fw.background_fraction(fr, world.bg_rows)
    assert bg.shape == (2, 2) and bg[0, 0] == bg[1, 1] == 0.0 and bg[0, 1] > 0.6


def test_flat_flag_is_a_threshold_on_the_background_fraction():
    bg = np.array([0.0, 0.04, 0.0501, 0.3])
    assert fw.flat_flags(bg).tolist() == [True, True, False, False]
    assert fw.flat_flags(bg, max_bg=0.5).tolist() == [True, True, True, True]


# ---------------------------------------------------------------- stuck on a flat wall (persistence)

def test_stuck_means_flat_for_the_whole_trailing_window():
    flat = np.array([[0, 1, 1, 1, 0, 1, 1, 1, 1]], dtype=bool)
    assert fw.stuck_flags(flat, 3).tolist() == [[False, False, False, True, False, False, False, True, True]]
    assert fw.stuck_flags(flat, 1).tolist() == flat.tolist()                     # window 1 = the plain per-frame flag
    assert fw.stuck_flags(flat, 10).sum() == 0                                  # a window longer than the rollout never fires


def test_stuck_never_looks_back_past_the_first_dreamed_frame():
    flat = np.ones((1, 6), dtype=bool)
    assert fw.stuck_flags(flat, 4).tolist() == [[False, False, False, True, True, True]]


# ---------------------------------------------------------------- collapse rate

def test_collapse_rate_counts_dream_frames_that_are_stuck_while_the_real_frame_is_not():
    E, L = 3, 600
    dream = np.ones((E, L), dtype=bool)
    real = np.zeros((E, L), dtype=bool)
    r = fw.collapse_rate(dream, real, window=8)
    assert r["rate"] == 1.0 and r["dream_stuck_share"] == 1.0 and r["real_stuck_share"] == 0.0
    r = fw.collapse_rate(dream, dream, window=8)                                # the real game is flat too: not a collapse
    assert r["rate"] == 0.0
    r = fw.collapse_rate(real, dream, window=8)                                 # a dream that is never flat cannot collapse
    assert r["rate"] == 0.0 and r["real_stuck_share"] == 1.0


def test_collapse_rate_only_looks_at_frames_64_to_512():
    """Frame numbers count from 1 (the first dreamed frame): t = 64 .. 512 inclusive."""
    E, L = 1, 520
    dream = np.zeros((E, L), dtype=bool)
    real = np.zeros((E, L), dtype=bool)
    dream[:, :63] = True                                       # frames 1..63: before the window of interest
    dream[:, 512:] = True                                      # frames 513..520: after it
    assert fw.collapse_rate(dream, real, window=1)["rate"] == 0.0
    dream[:, 63] = True                                        # frame 64 is inside
    assert fw.collapse_rate(dream, real, window=1)["rate"] == pytest.approx(1 / 449)
    dream[:, 511] = True                                       # frame 512 is inside too
    assert fw.collapse_rate(dream, real, window=1)["rate"] == pytest.approx(2 / 449)


def test_collapse_rate_reports_the_per_episode_rates_for_the_bootstrap():
    dream = np.zeros((4, 520), dtype=bool)
    real = np.zeros((4, 520), dtype=bool)
    dream[1, 100:200] = True                                   # one episode stuck for 100 frames
    r = fw.collapse_rate(dream, real, window=1)
    assert r["per_episode"].shape == (4,) and r["per_episode"][1] == pytest.approx(100 / 449) and r["per_episode"][0] == 0
    assert r["rate"] == pytest.approx(r["per_episode"].mean())


def test_bootstrap_interval_brackets_the_mean_and_is_reproducible():
    x = np.array([0.0, 0.1, 0.5, 0.2, 0.0, 0.3, 0.1, 0.0])
    lo, hi = fw.bootstrap_ci(x, seed=1)
    assert lo < x.mean() < hi and (lo, hi) == fw.bootstrap_ci(x, seed=1)
    lo2, hi2 = fw.bootstrap_ci(np.full(8, 0.2), seed=1)
    assert lo2 == hi2 == pytest.approx(0.2)


def test_run_lengths_of_flat_spells():
    flat = np.array([[1, 1, 0, 1, 1, 1, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=bool)
    assert sorted(fw.run_lengths(flat)) == [1, 2, 3]


def test_collapse_rate_of_a_rollout_shorter_than_the_window_of_interest_is_nan_without_warnings():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = fw.collapse_rate(np.ones((2, 40), dtype=bool), np.zeros((2, 40), dtype=bool), window=8)
    assert np.isnan(r["rate"]) and np.isnan(r["dream_stuck_share"])
