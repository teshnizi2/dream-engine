"""Game v3: self-localising walls (docs/PREREG-v3.md).

v2's walls inside one colour region are identical, and so are its N and S (and E and W) faces, so a close-up of a wall says nothing
about where the player is. v3 adds, and only adds:

  * a low-frequency colour tint on every wall that depends on the world position of the hit point (any wall view, even a
    close-up, tells where along the wall you are), keeping each region's base colour recognisable;
  * four distinct shades for the four wall-face directions (so a wall view also says which way you are facing).

No new high-frequency texture (v2's finding: stripes cost error), the same map, physics, floor, ceiling, fog and landmarks. The
Python and JavaScript twins stay bit-exact (tests/parity.test.mjs); the v2 game stays reproducible bit for bit.
"""
import hashlib
import json
import pathlib

import numpy as np
import pytest

from game import raycaster as rc

HERE = pathlib.Path(rc.__file__).parent
SPEC_V2, SPEC_V3 = HERE / "spec_v2.json", HERE / "spec_v3.json"

# frames of the v2 game rendered by the code at the v2-final tag (see make_golden below); v3 code must not change them
GOLDEN_RANDOM_POSES = "8a42dac792fe3b7accbd26a91e2a872d9f75767e28f4cb6a5f75ae05d06d93c2"
GOLDEN_WALK = "7a851f58c59e63ac270a296e481844f8d3a10f6c72172db6985baf6467615f9e"


def make_golden(world):
    rng = np.random.default_rng(2024)
    free = world.free_cells()
    M = 120
    idx = rng.integers(0, len(free), M)
    st = rc.state_from_pose(free[idx, 0] + rng.uniform(0.0, 1.0, M), free[idx, 1] + rng.uniform(0.0, 1.0, M), rng.uniform(0, 2 * np.pi, M))
    random_poses = world.render(st)
    st = rc.state_from_pose(np.array([2.5]), np.array([7.5]), np.array([0.0]))
    acts = np.array([[1, 0, 0]] * 20 + [[1, 0, 2]] * 30 + [[0, 2, 1]] * 20 + [[2, 0, 0]] * 10)
    frames = [world.render(st)]
    for a in acts:
        st = world.step(st, a[None])
        frames.append(world.render(st))
    return hashlib.sha256(random_poses.tobytes()).hexdigest(), hashlib.sha256(np.concatenate(frames).tobytes()).hexdigest()


@pytest.fixture(scope="module")
def spec3():
    return rc.load_spec(SPEC_V3)


@pytest.fixture(scope="module")
def world3(spec3):
    return rc.World(spec3)


@pytest.fixture(scope="module")
def world2():
    return rc.World(rc.load_spec(SPEC_V2))


def S(x, y, deg=0.0):
    return rc.state_from_pose(np.array([x]), np.array([y]), np.radians(np.array([deg])))


def exposed_faces(world):
    """(cx, cy, face) of every wall-cell face that borders an open cell: the faces a player can ever see."""
    g, N = world.grid, world.N
    out = []
    for cy in range(N):
        for cx in range(N):
            if g[cy, cx] == 0:
                continue
            for face, (dx, dy) in ((rc.FACE_W, (-1, 0)), (rc.FACE_E, (1, 0)), (rc.FACE_N, (0, -1)), (rc.FACE_S, (0, 1))):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < N and 0 <= ny < N and g[ny, nx] == 0:
                    out.append((cx, cy, face))
    return out


# ---------------------------------------------------------------- v2 stays exactly v2

def test_the_v3_code_still_renders_the_v2_game_bit_for_bit(world2):
    assert make_golden(world2) == (GOLDEN_RANDOM_POSES, GOLDEN_WALK)


def test_v3_spec_only_adds_the_signature_layer_to_v2(spec3):
    v2 = rc.load_spec(SPEC_V2)
    assert spec3["game_version"] == 3 and "game_version" not in v2 and "tint" in spec3 and "tint" not in v2
    for k, v in v2.items():
        if k != "walls":
            assert spec3[k] == v, k                       # map, physics, floor, ceiling, fog, mip, start, actions: untouched
    for i, w in v2["walls"].items():
        assert {k: v for k, v in spec3["walls"][i].items() if k != "tint_gain"} == w, i      # colours and textures: untouched


def test_make_spec_regenerates_both_frozen_specs_exactly():
    from game import make_spec
    assert json.loads(json.dumps(make_spec.build(version=3))) == json.loads(SPEC_V3.read_text())
    assert json.loads(json.dumps(make_spec.build(version=2))) == json.loads(SPEC_V2.read_text())


def test_shipped_spec_is_one_of_the_two_frozen_games():
    shipped = (HERE / "spec.json").read_bytes()
    assert shipped in (SPEC_V2.read_bytes(), SPEC_V3.read_bytes())


# ---------------------------------------------------------------- which face did the ray hit?

def hit_cell(hit, face, i=0):
    """The wall cell a ray hit, recovered from its hit point and face (the hit point lies exactly on that face of that cell)."""
    hx, hy = float(hit.hx[0, i]), float(hit.hy[0, i])
    if face == rc.FACE_W:
        return int(round(hx)), int(np.floor(hy))
    if face == rc.FACE_E:
        return int(round(hx)) - 1, int(np.floor(hy))
    if face == rc.FACE_N:
        return int(np.floor(hx)), int(round(hy))
    return int(np.floor(hx)), int(round(hy)) - 1


@pytest.mark.parametrize("deg,face", [(0.0, rc.FACE_W), (180.0, rc.FACE_E), (90.0, rc.FACE_N), (-90.0, rc.FACE_S)])
def test_a_ray_reports_which_face_of_the_wall_cell_it_hit(world3, deg, face):
    """Heading 0 looks east (+x) and hits the west face of the wall cell ahead; 90 degrees looks south (+y) and hits its north face."""
    hit = world3.cast_hits(S(8.5, 7.5, deg), np.array([0.0]))
    assert int(hit.face[0, 0]) == face
    cx, cy = hit_cell(hit, face)
    assert world3.grid[cy, cx] == int(hit.wid[0, 0]) > 0


def test_hit_points_and_faces_are_consistent_for_all_rays_of_random_views(world3):
    rng = np.random.default_rng(8)
    free = world3.free_cells()
    idx = rng.integers(0, len(free), 12)
    st = rc.state_from_pose(free[idx, 0] + rng.uniform(0.2, 0.8, 12), free[idx, 1] + rng.uniform(0.2, 0.8, 12), rng.uniform(0, 2 * np.pi, 12))
    hit = world3.cast_hits(st, world3.camx)
    for b in range(12):
        for i in range(0, 64, 7):
            one = rc.State(*[a[b:b + 1] for a in st])
            h = world3.cast_hits(one, world3.camx[i:i + 1])
            face = int(h.face[0, 0])
            assert face == int(hit.face[b, i]) and float(h.hx[0, 0]) == float(hit.hx[b, i])
            cx, cy = hit_cell(h, face)
            assert world3.grid[cy, cx] == int(h.wid[0, 0]) > 0, (b, i, face, cx, cy)


def test_cast_and_cast_hits_agree_on_the_ray_itself(world3):
    rng = np.random.default_rng(3)
    free = world3.free_cells()
    idx = rng.integers(0, len(free), 20)
    st = rc.state_from_pose(free[idx, 0] + rng.uniform(0.2, 0.8, 20), free[idx, 1] + rng.uniform(0.2, 0.8, 20), rng.uniform(0, 2 * np.pi, 20))
    perp, side, wid, u = world3.cast(st, world3.camx)
    hit = world3.cast_hits(st, world3.camx)
    assert np.array_equal(perp, hit.perp) and np.array_equal(side, hit.side) and np.array_equal(wid, hit.wid) and np.array_equal(u, hit.u)


# ---------------------------------------------------------------- four distinct face shades

def test_the_four_face_directions_have_distinct_shades(spec3):
    shades = spec3["tint"]["face_shade"]
    assert len(shades) == 4 and len(set(shades)) == 4
    s = sorted(shades)
    assert min(b - a for a, b in zip(s, s[1:])) >= 0.08, "adjacent shades must differ by at least 8% so that a wall view says which way it faces"
    assert max(shades) <= 1.0 and min(shades) >= 0.5, "dark faces must stay readable"


def test_one_block_looks_different_from_each_of_its_four_sides(world3):
    """The amber pillar is a free-standing block: seen from the west, east, north and south at the same distance it is told apart by
    brightness alone (the pillar keeps its exact amber look, so no tint)."""
    cols = []
    for (x, y, deg) in ((5.5, 7.5, 0.0), (9.5, 7.5, 180.0), (7.5, 5.5, 90.0), (7.5, 9.5, -90.0)):      # 2 cells from the pillar at (7, 7)
        cols.append(world3.render(S(x, y, deg))[0].astype(float)[24:40, 31:33].reshape(-1, 3).mean(0))
    lum = np.array([c.sum() for c in cols])
    order = np.argsort(-lum)
    assert order.tolist() == [0, 1, 2, 3], lum              # W face brightest, then E, N, S: the face shades, in order
    assert np.diff(-np.sort(-lum)).max() < -8, lum           # and each step is large enough to see


# ---------------------------------------------------------------- every face has its own colour

def test_every_visible_wall_face_has_a_unique_colour_signature(world3):
    """For each wall type, the colour of every exposed face (shade x (base + tint at the face centre)) differs from all others
    by at least 3 grey levels: faces of one type are told apart even in a flat close-up."""
    by_type = {}
    for cx, cy, face in exposed_faces(world3):
        by_type.setdefault(int(world3.grid[cy, cx]), []).append(world3.face_colour(cx, cy, face))
    assert len(by_type) >= 7
    for wid, cols in by_type.items():
        c = np.array(cols)
        d = np.linalg.norm(c[:, None] - c[None], axis=-1) + np.eye(len(c)) * 1e9
        assert d.min() >= 3.0, (wid, len(c), d.min())


def test_neighbouring_cells_along_a_wall_differ_clearly_enough_to_localise_to_one_cell(world3):
    """The hall's north wall (y = 5, cells x = 5..10): the tint moves at least 3 grey levels from one cell to the next."""
    cols = np.array([world3.face_colour(cx, 4, rc.FACE_S) for cx in range(5, 11)])
    steps = np.linalg.norm(np.diff(cols, axis=0), axis=1)
    assert steps.min() >= 3.0, steps


def test_each_regions_base_colour_stays_recognisable(world3, spec3):
    """Tint and shading move a colour, they never turn it into another region's colour: every exposed face is closer (in colour) to its
    own wall type's base colour, shaded the same way, than to any other type's."""
    base = {int(i): np.array(w["rgb"], dtype=float) for i, w in spec3["walls"].items()}
    shade = spec3["tint"]["face_shade"]
    for cx, cy, face in exposed_faces(world3):
        wid = int(world3.grid[cy, cx])
        col = world3.face_colour(cx, cy, face) / shade[face]                      # undo the face shade
        dist = {i: np.linalg.norm(col - b) for i, b in base.items()}
        assert min(dist, key=dist.get) == wid, (cx, cy, face, wid, dist)


# ---------------------------------------------------------------- low frequency only

def test_the_tint_is_smooth_in_the_world_and_has_no_steps_at_cell_boundaries(world3):
    xs = np.linspace(0.0, 16.0, 1601)
    for y in (3.0, 7.37, 12.0):
        t = world3.tint_offsets(xs, np.full_like(xs, y))
        assert t.shape == (1601, 3)
        slope = np.abs(np.diff(t, axis=0)).max()                                # per 0.01 cell
        assert slope < 0.15, slope                                                # <= 15 grey levels per cell, no jumps anywhere
    ys = np.linspace(0.0, 16.0, 1601)
    t = world3.tint_offsets(np.full_like(ys, 9.5), ys)
    assert np.abs(np.diff(t, axis=0)).max() < 0.15


def test_the_tint_is_bounded_by_its_amplitude_and_uses_all_three_channels_differently(world3, spec3):
    xs, ys = np.meshgrid(np.linspace(0, 16, 97), np.linspace(0, 16, 97))
    t = world3.tint_offsets(xs.ravel(), ys.ravel())
    amp = spec3["tint"]["amplitude"]
    assert t.shape == (97 * 97, 3) and np.abs(t).max() <= amp + 1e-9 and np.abs(t).max() > 0.9 * amp
    c = np.corrcoef(t.T)
    assert np.abs(c[np.triu_indices(3, 1)]).max() < 0.5, "three different waves, not one tint copied to every channel"


def test_a_close_up_of_a_wall_adds_no_pixel_scale_texture_over_v2(world2, world3):
    """The tint varies by well under a grey level between neighbouring pixels: v3 close-ups have the same fine structure as v2's."""
    def grad(w, st):
        f = w.render(st)[0].astype(float)
        return np.abs(np.diff(f, axis=1)).mean()
    for pose in (S(8.5, 5.5, -90.0), S(7.5, 1.5, -90.0), S(13.5, 7.5, 0.0), S(2.5, 7.5, 0.0)):
        assert grad(world3, pose) <= grad(world2, pose) + 0.35


# ---------------------------------------------------------------- the render follows the spec

def test_a_far_flat_wall_is_rendered_exactly_as_the_spec_says(world3, spec3):
    """Beyond 7 cells the wall texture has faded to its mean colour, so a pixel is  fog + ((rgb + gain * tint) * shade - fog) * f."""
    st = S(1.5, 12.5, 0.0)                                               # the south corridor, looking east at the ochre end wall 13.5 cells away
    hit = world3.cast_hits(st, world3.camx)
    i = 32
    wid, face, perp = int(hit.wid[0, i]), int(hit.face[0, i]), float(hit.perp[0, i])
    assert wid == 8 and face == rc.FACE_W and perp > 7.0
    w = spec3["walls"][str(wid)]
    kbar = w["p"][4]
    base = np.array(w["rgb"]) + (np.array(w["rgb2"]) - np.array(w["rgb"])) * kbar
    off = world3.tint_offsets(hit.hx[0, i], hit.hy[0, i]) * w["tint_gain"]
    c = (base + off) * spec3["tint"]["face_shade"][face]
    f = 1.0 / (1.0 + (perp * spec3["fog_k"]) ** 2)
    fog = np.array(spec3["fog_color"], dtype=float)
    expect = np.clip(np.floor(fog + (c - fog) * f + 0.5), 0, 255)
    assert np.array_equal(world3.render(st)[0, 32, i].astype(float), expect)


# ---------------------------------------------------------------- landmarks and the rest of the look

def test_landmarks_keep_their_look(world3):
    f = world3.render(S(5.5, 7.5, 0.0))[0]                                     # amber pillar
    r, g, b = f[28:36, 30:34].reshape(-1, 3).mean(0)
    assert r > 150 and r > g > b
    f = world3.render(S(13.5, 7.5, 0.0))[0]                                    # red brick room
    r, g, b = f[28:36, 30:34].reshape(-1, 3).mean(0)
    assert r > 1.5 * g and r > 1.5 * b
    f = world3.render(S(8.5, 6.5, 180.0))[0]                                   # blue wall
    r, g, b = f[28:36, 28:36].reshape(-1, 3).mean(0)
    assert b > r + 20 and b > g
    f = world3.render(S(7.5, 1.5, -90.0))[0].astype(float)                     # window gallery stays striped
    row = f[32, :, :].sum(-1)
    assert row.max() - row.min() > 200 and (np.abs(np.diff(row)) > 100).sum() >= 4


def test_floor_ceiling_and_dynamics_are_identical_to_v2(world2, world3):
    assert np.array_equal(world2.bg_rows, world3.bg_rows)
    rng = np.random.default_rng(1)
    free = world2.free_cells()
    idx = rng.integers(0, len(free), 32)
    s2 = s3 = rc.state_from_pose(free[idx, 0] + rng.uniform(0.25, 0.75, 32), free[idx, 1] + rng.uniform(0.25, 0.75, 32), rng.uniform(0, 2 * np.pi, 32))
    for _ in range(200):
        a = np.stack([rng.integers(0, 3, 32), rng.integers(0, 3, 32), rng.integers(0, 3, 32)], 1)
        s2, s3 = world2.step(s2, a), world3.step(s3, a)
    for u, v in zip(s2, s3):
        assert np.array_equal(u, v)


def test_geometry_is_the_same_so_the_flat_wall_flags_of_real_frames_are_the_same_in_v2_and_v3(world2, world3):
    """The background (floor/ceiling) pixels of a v3 frame are exactly those of the v2 frame at the same pose: the flat-wall metric
    sees the same real game in both versions."""
    rng = np.random.default_rng(5)
    free = world2.free_cells()
    idx = rng.integers(0, len(free), 60)
    st = rc.state_from_pose(free[idx, 0] + rng.uniform(0.1, 0.9, 60), free[idx, 1] + rng.uniform(0.1, 0.9, 60), rng.uniform(0, 2 * np.pi, 60))
    bg = world2.bg_rows[None, :, None, :]
    from model import flatwall as fw
    b2 = fw.background_fraction(world2.render(st), world2.bg_rows)
    b3 = fw.background_fraction(world3.render(st), world3.bg_rows)
    assert np.array_equal(fw.flat_flags(b2), fw.flat_flags(b3))
    assert np.abs(b2 - b3).max() < 0.02
