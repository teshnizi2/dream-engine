"""The scripted tour (used by the autoplay button and the video) must walk through the landmark regions of the map
without getting stuck, and be reproducible from the committed json."""
import json
import pathlib

import numpy as np

from game import raycaster as rc

ROOT = pathlib.Path(__file__).resolve().parents[1]


def replay():
    tour = json.loads((ROOT / "site/js/tour.json").read_text())
    w = rc.World()
    st = rc.state_from_spec_start(w.spec)
    xs, ys = [], []
    for a in tour["actions"]:
        st = w.step(st, np.array([a], dtype=np.int64))
        xs.append(float(st.x[0]))
        ys.append(float(st.y[0]))
    return tour, np.array(xs), np.array(ys)


def test_tour_is_long_enough_for_a_20_second_clip_and_loops_back_to_the_pillar_room():
    tour, xs, ys = replay()
    assert 480 <= len(tour["actions"]) <= 1100
    assert 5 <= xs[-1] <= 7 and 7 <= ys[-1] <= 8, "ends in the hall doorway, looking at the pillar"


def test_tour_visits_the_pillar_hall_the_window_gallery_and_the_red_room():
    _, xs, ys = replay()
    cells = {(int(x), int(y)) for x, y in zip(xs, ys)}
    assert any(5 <= c <= 10 and 5 <= r <= 9 for c, r in cells), "hall with the amber pillar"
    assert any(r == 1 and 6 <= c <= 13 for c, r in cells), "window gallery (north wall)"
    assert any(12 <= c <= 14 and 6 <= r <= 9 for c, r in cells), "red-brick room"
    assert any(11 <= c <= 14 and 11 <= r <= 14 for c, r in cells), "ochre room"
    assert any(1 <= c <= 4 and 11 <= r <= 14 for c, r in cells), "purple room"
    assert any(c == 2 and 4 <= r <= 10 for c, r in cells), "green west corridor"


def test_tour_keeps_moving():
    _, xs, ys = replay()
    moved = np.hypot(np.diff(xs), np.diff(ys)) > 1e-9
    assert moved.mean() > 0.55, f"agent translates on only {moved.mean():.2f} of steps"
    assert np.hypot(xs[-1] - xs[0], ys[-1] - ys[0]) > 2.0, "ends away from the start"
