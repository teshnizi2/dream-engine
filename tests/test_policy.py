"""Tests for the exploration policy and the on-the-fly trajectory generator (data/)."""
import numpy as np
import pytest

from game import raycaster as rc
from data import policy as pol
from data import loader


@pytest.fixture(scope="module")
def world():
    return rc.World()


def run_policy(world, n_envs=64, steps=800, seed=0, respawn=True):
    rng = np.random.default_rng(seed)
    p = pol.ExplorationPolicy(world, n_envs, rng, respawn_prob=1 / 300 if respawn else 0.0)
    st = p.random_states(n_envs)
    acts, xs, ys = [], [], []
    for _ in range(steps):
        a = p.act(st)
        st = world.step(st, a)
        st = p.maybe_respawn(st)
        acts.append(a)
        xs.append(st.x.copy())
        ys.append(st.y.copy())
    return np.stack(acts), np.stack(xs), np.stack(ys)


def test_actions_are_valid_triples(world):
    a, _, _ = run_policy(world, steps=50)
    assert a.shape == (50, 64, 3) and a.min() >= 0 and a.max() <= 2


def test_actions_are_sticky(world):
    a, _, _ = run_policy(world, steps=400, respawn=False)
    for f in range(3):
        changes = (a[1:, :, f] != a[:-1, :, f]).mean()
        assert changes < 0.30, f"factor {f} changes {changes:.2f} of steps (should be sticky)"


def test_policy_is_biased_towards_forward_and_turns(world):
    a, _, _ = run_policy(world, steps=600)
    fwd = (a[..., 0] == 1).mean()
    back = (a[..., 0] == 2).mean()
    turn = (a[..., 2] != 0).mean()
    strafe = (a[..., 1] != 0).mean()
    assert fwd > 0.45 and fwd > 4 * back
    assert 0.2 < turn < 0.8
    assert 0.02 < strafe < 0.25


def test_policy_does_not_push_into_walls(world):
    rng = np.random.default_rng(0)
    p = pol.ExplorationPolicy(world, 64, rng, respawn_prob=0.0)
    st = p.random_states(64)
    pushing, moved = [], []
    for _ in range(600):
        a = p.act(st)
        st2 = world.step(st, a)
        m = (np.abs(st2.x - st.x) + np.abs(st2.y - st.y)) > 1e-9
        pushing.append((a[:, 0] == rc.MOVE_FWD) & ~m)
        moved.append(m)
        st = st2
    assert np.mean(pushing) < 0.03, f"pressing forward into a wall on {np.mean(pushing):.3f} of steps"
    assert np.mean(moved) > 0.65, f"agents translate on {np.mean(moved):.2f} of steps"


def test_policy_covers_the_map_evenly(world):
    _, xs, ys = run_policy(world, n_envs=128, steps=1500)
    cells = np.floor(ys).astype(int) * 16 + np.floor(xs).astype(int)
    counts = np.bincount(cells.ravel(), minlength=256).astype(float)
    free = (world.grid.ravel() == 0)
    c = counts[free]
    assert (c > 0).all(), "every open cell is visited"
    p = c / c.sum()
    evenness = -(p * np.log(p)).sum() / np.log(len(p))
    assert evenness > 0.9, f"normalised entropy of visits {evenness:.3f}"


def test_respawn_teleports_to_open_cells(world):
    rng = np.random.default_rng(3)
    p = pol.ExplorationPolicy(world, 500, rng, respawn_prob=1.0)
    st = p.random_states(500)
    st2 = p.maybe_respawn(st)
    cells = world.grid[np.floor(st2.y).astype(int), np.floor(st2.x).astype(int)]
    assert (cells == 0).all()
    assert np.allclose(np.hypot(st2.dx, st2.dy), 1.0)


def test_training_batch_has_expected_shapes_and_consistent_actions(world):
    src = loader.TrajectorySource(world, batch_size=16, seq_len=5, seed=0)
    b = src.next_batch()
    assert b["frames"].shape == (16, 5, 64, 64, 3) and b["frames"].dtype == np.uint8
    assert b["actions"].shape == (16, 4, 3) and b["actions"].dtype == np.int64
    # frame t+1 is exactly the game's response to action t
    st = rc.State(*[b["poses"][:, 0, i] for i in range(4)])
    for t in range(4):
        st = world.step(st, b["actions"][:, t])
        assert np.array_equal(world.render(st), b["frames"][:, t + 1])


def test_generator_is_deterministic_given_seed(world):
    a = loader.TrajectorySource(world, 8, 5, seed=7).next_batch()
    b = loader.TrajectorySource(world, 8, 5, seed=7).next_batch()
    assert np.array_equal(a["frames"], b["frames"]) and np.array_equal(a["actions"], b["actions"])


def test_eval_trajectories_are_fixed_and_long(world):
    t = loader.eval_trajectories(world, n=4, length=40, seed=123)
    assert t["frames"].shape == (4, 40, 64, 64, 3) and t["actions"].shape == (4, 39, 3)
    t2 = loader.eval_trajectories(world, n=4, length=40, seed=123)
    assert np.array_equal(t["frames"], t2["frames"])
