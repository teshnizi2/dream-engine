"""Tests for the Python raycaster (the reference implementation of the game).

Written first (TDD): they define what the game must do. The JS twin is checked against this
implementation in tests/parity.test.mjs.
"""
import math
from collections import deque

import numpy as np
import pytest

from game import raycaster as rc


@pytest.fixture(scope="module")
def spec():
    return rc.load_spec()


@pytest.fixture(scope="module")
def world(spec):
    return rc.World(spec)


def S(x, y, deg=0.0):
    """Single-env state at (x, y) facing `deg` degrees (0 = east, 90 = south)."""
    return rc.state_from_pose(np.array([x]), np.array([y]), np.radians(np.array([deg])))


def act(move=0, strafe=0, turn=0):
    return np.array([[move, strafe, turn]], dtype=np.int64)


# ---------------------------------------------------------------- map

def test_map_is_16x16_with_closed_border(world):
    g = world.grid
    assert g.shape == (16, 16)
    assert (g[0, :] > 0).all() and (g[-1, :] > 0).all()
    assert (g[:, 0] > 0).all() and (g[:, -1] > 0).all()


def test_all_open_cells_are_reachable_and_map_has_loops(world):
    g = world.grid
    open_cells = {(c, r) for r in range(16) for c in range(16) if g[r, c] == 0}
    assert 90 <= len(open_cells) <= 160, "map should be roomy but not empty"
    start = next(iter(open_cells))
    seen, q = {start}, deque([start])
    while q:
        c, r = q.popleft()
        for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (c + dc, r + dr)
            if n in open_cells and n not in seen:
                seen.add(n)
                q.append(n)
    assert seen == open_cells
    edges = sum(1 for (c, r) in open_cells for dc, dr in ((1, 0), (0, 1)) if (c + dc, r + dr) in open_cells)
    assert edges - len(open_cells) + 1 >= 3, "need at least 3 independent loops"


def test_landmark_wall_types_exist_in_the_map(world, spec):
    names = {int(k): v["name"] for k, v in spec["walls"].items()}
    present = {names[int(i)] for i in np.unique(world.grid) if i > 0}
    for landmark in ("amber_pillar", "blue_door", "red_brick", "window_stripes"):
        assert landmark in present, landmark


def test_start_pose_is_in_open_space_looking_down_a_line_of_sight_to_the_pillar(world, spec):
    s = spec["start"]
    assert world.grid[int(s["y"]), int(s["x"])] == 0
    assert abs(math.hypot(s["dx"], s["dy"]) - 1.0) < 1e-12
    frame = world.render(rc.state_from_pose(np.array([s["x"]]), np.array([s["y"]]), np.array([math.atan2(s["dy"], s["dx"])])))
    centre = frame[0, 24:40, 30:34].reshape(-1, 3).mean(0)
    assert centre[0] > centre[2] + 40, "amber pillar should be framed at the centre of the first frame"


# ---------------------------------------------------------------- rendering

def test_render_shape_dtype_and_not_blank(world):
    f = world.render(S(8.5, 7.5, 90.0))
    assert f.shape == (1, 64, 64, 3) and f.dtype == np.uint8
    assert f.std() > 10


def test_render_is_deterministic(world):
    a = world.render(S(2.5, 5.5, 30.0))
    b = world.render(S(2.5, 5.5, 30.0))
    assert np.array_equal(a, b)


def test_batch_render_equals_single_renders(world):
    rng = np.random.default_rng(0)
    free = world.free_cells()
    idx = rng.integers(0, len(free), 12)
    xs = free[idx, 0] + rng.uniform(0.3, 0.7, 12)
    ys = free[idx, 1] + rng.uniform(0.3, 0.7, 12)
    ang = rng.uniform(0, 2 * np.pi, 12)
    batch = world.render(rc.state_from_pose(xs, ys, ang))
    for i in range(12):
        single = world.render(rc.state_from_pose(xs[i:i + 1], ys[i:i + 1], ang[i:i + 1]))
        assert np.array_equal(batch[i], single[0]), i


def test_fog_dims_a_distant_wall(world):
    # same stone wall (hall north wall, col 8) seen from 1.1 cells away and from 5.1 cells away
    near = world.render(S(8.5, 5.5, -90.0))[0, 22:26, 30:34].astype(float).sum(-1).mean()   # wall fills the view
    far = world.render(S(8.5, 9.5, -90.0))[0, 27:30, 30:34].astype(float).sum(-1).mean()    # wall spans rows 25-39
    assert near > far * 1.15


def test_amber_pillar_looks_amber(world):
    f = world.render(S(5.5, 7.5, 0.0))[0]            # looking east at the pillar in the hall
    r, g, b = f[28:36, 30:34].reshape(-1, 3).mean(0)
    assert r > 150 and r > g > b


def test_red_room_looks_red(world):
    f = world.render(S(13.5, 7.5, 0.0))[0]           # east wall of the red room
    r, g, b = f[28:36, 30:34].reshape(-1, 3).mean(0)
    assert r > 1.5 * g and r > 1.5 * b


def test_blue_doorway_wall_looks_blue(world):
    f = world.render(S(8.5, 6.5, 180.0))[0]          # hall west wall, blue
    r, g, b = f[28:36, 28:36].reshape(-1, 3).mean(0)
    assert b > r + 20 and b > g


def test_window_gallery_wall_is_striped(world):
    f = world.render(S(7.5, 1.5, -90.0))[0].astype(float)   # north wall of the gallery, 1 cell away
    row = f[32, :, :].sum(-1)
    assert row.max() - row.min() > 200, "bright windows against dark frames"
    assert (np.abs(np.diff(row)) > 100).sum() >= 4, "several window edges across the view"


def test_fine_texture_fades_to_a_flat_colour_with_distance(world):
    """Pixel-sized patterns on far walls alias into noise that no small network can predict, so textures are
    pre-filtered: full detail up close, the wall's mean colour beyond ~7 cells."""
    # ochre end wall of the south corridor (2x2 blocks with mortar lines): 1.5 cells away vs 13.5 cells away
    near = world.render(S(13.5, 12.5, 0.0))[0].astype(float)[8:56, 30:34].sum(-1)
    far = world.render(S(1.5, 12.5, 0.0))[0].astype(float)[31:34, 30:34].sum(-1)           # the wall is ~5 px tall here
    assert near.std(axis=0).max() > 12, "visible mortar lines up close"
    assert far.std() < 3, "no texture left at 13 cells"


def test_window_stripes_are_wide_enough_to_be_learnable(world):
    """Stripes were 5% of the pixels and 65% of a trained model's squared error; a pane should span several pixels."""
    f = world.render(S(7.5, 2.5, -90.0))[0].astype(float)   # 1.5 cells from the window wall: panes ~10 px wide
    row = f[32, :, :].sum(-1)
    bright = row > 0.5 * (row.max() + row.min())
    runs, cur = [], 0
    for b in bright:
        if b:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    runs = [r for r in runs if r > 1]
    assert runs and min(runs) >= 5, runs


# ---------------------------------------------------------------- dynamics

def test_noop_leaves_state_exactly_unchanged(world):
    s = S(8.5, 7.5, 33.0)
    n = world.step(s, act())
    for a, b in zip(s, n):
        assert np.array_equal(a, b)


def test_forward_moves_move_speed_along_heading(world, spec):
    s = S(8.5, 8.5, 0.0)
    n = world.step(s, act(move=1))
    assert n.x[0] == pytest.approx(8.5 + spec["move_speed"], abs=1e-12)
    assert n.y[0] == pytest.approx(8.5, abs=1e-12)


def test_back_is_the_opposite_of_forward(world):
    s = S(8.5, 8.5, 40.0)
    f = world.step(s, act(move=1))
    b = world.step(s, act(move=2))
    assert (f.x[0] - s.x[0]) == pytest.approx(-(b.x[0] - s.x[0]), abs=1e-12)
    assert (f.y[0] - s.y[0]) == pytest.approx(-(b.y[0] - s.y[0]), abs=1e-12)


def test_strafe_right_moves_to_the_viewers_right(world, spec):
    s = S(8.5, 8.5, 0.0)                              # facing east: viewer's right is south (+y)
    n = world.step(s, act(strafe=2))
    assert n.y[0] == pytest.approx(8.5 + spec["strafe_speed"], abs=1e-12)
    assert n.x[0] == pytest.approx(8.5, abs=1e-12)
    n = world.step(s, act(strafe=1))
    assert n.y[0] == pytest.approx(8.5 - spec["strafe_speed"], abs=1e-12)


def test_turn_right_rotates_by_turn_deg_and_keeps_unit_length(world, spec):
    s = S(8.5, 8.5, 10.0)
    n = world.step(s, act(turn=2))
    d = math.degrees(math.atan2(n.dy[0], n.dx[0]) - math.atan2(s.dy[0], s.dx[0]))
    assert d == pytest.approx(spec["turn_deg"], abs=1e-9)
    assert math.hypot(n.dx[0], n.dy[0]) == pytest.approx(1.0, abs=1e-12)
    n = world.step(s, act(turn=1))
    d = math.degrees(math.atan2(n.dy[0], n.dx[0]) - math.atan2(s.dy[0], s.dx[0]))
    assert d == pytest.approx(-spec["turn_deg"], abs=1e-9)


def test_turning_and_moving_combine_in_one_step(world, spec):
    s = S(8.5, 8.5, 0.0)
    n = world.step(s, act(move=1, turn=2))             # turn first, then move along the new heading
    assert n.dy[0] > 0
    assert math.hypot(n.x[0] - 8.5, n.y[0] - 8.5) == pytest.approx(spec["move_speed"], abs=1e-12)
    assert n.y[0] > 8.5


def test_a_wall_stops_the_agent_before_the_wall_cell(world, spec):
    s = S(10.2, 7.5, 0.0)                               # hall east wall (col 11) is 0.8 ahead
    for _ in range(30):
        s = world.step(s, act(move=1))
    assert s.x[0] < 11.0 - spec["radius"] + 1e-9
    assert s.x[0] > 10.5


def test_agent_slides_along_a_wall(world, spec):
    s = S(10.5, 6.0, 45.0)                              # heading south-east into the east wall
    y0 = s.y[0]
    for _ in range(10):
        s = world.step(s, act(move=1))
    assert s.x[0] < 11.0 - spec["radius"] + 1e-9
    assert s.y[0] > y0 + 0.3, "still moves along the wall"


def test_random_walks_never_leave_open_cells(world):
    rng = np.random.default_rng(1)
    free = world.free_cells()
    B = 64
    idx = rng.integers(0, len(free), B)
    s = rc.state_from_pose(free[idx, 0] + rng.uniform(0.25, 0.75, B), free[idx, 1] + rng.uniform(0.25, 0.75, B),
                           rng.uniform(0, 2 * np.pi, B))
    for _ in range(600):
        a = np.stack([rng.integers(0, 3, B), rng.integers(0, 3, B), rng.integers(0, 3, B)], 1)
        s = world.step(s, a)
        cells = world.grid[np.floor(s.y).astype(int), np.floor(s.x).astype(int)]
        assert (cells == 0).all()
    assert np.allclose(np.hypot(s.dx, s.dy), 1.0, atol=1e-12)


def test_ray_distance_ahead_reports_wall_distance(world):
    d = world.distance_ahead(S(10.2, 7.5, 0.0))
    assert d[0] == pytest.approx(0.8, abs=1e-9)
