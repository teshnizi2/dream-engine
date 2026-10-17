"""Exploration policy: sticky random actions, biased towards forward and turns, with wall avoidance
and random respawns, vectorised over a batch of independent games.

Each of the three action factors (move, strafe, turn) holds its current value for a geometric
number of steps, then resamples. If the agent walks into a wall it turns towards the more open
side. Occasional random respawns spread the visits evenly over the map.
"""
import numpy as np

from game import raycaster as rc


class ExplorationPolicy:
    # (probabilities, mean duration in steps) per factor value
    MOVE = ([0.12, 0.76, 0.12], [6.0, 18.0, 6.0])
    TURN = ([0.45, 0.275, 0.275], [8.0, 7.0, 7.0])
    STRAFE = ([0.88, 0.06, 0.06], [5.0, 5.0, 5.0])
    PROBE = np.array([-0.9, -0.5, 0.0, 0.5, 0.9])    # camera-plane offsets of the rays used for avoidance

    def __init__(self, world, n_envs, rng, respawn_prob=1 / 250, avoid_dist=0.7):
        self.world, self.n, self.rng = world, n_envs, rng
        self.respawn_prob = respawn_prob
        self.avoid_dist = avoid_dist
        self.free = world.free_cells()
        self.reset()

    def reset(self, mask=None):
        """Forget sticky state (all envs, or those in `mask`)."""
        if mask is None:
            self.cur = np.zeros((self.n, 3), dtype=np.int64)
            self.left = np.zeros((self.n, 3), dtype=np.int64)
        else:
            self.left[mask] = 0

    def _resample(self, f, mask):
        probs, means = (self.MOVE, self.STRAFE, self.TURN)[f]
        k = mask.sum()
        if k == 0:
            return
        val = self.rng.choice(3, size=k, p=probs)
        mean = np.array(means)[val]
        dur = self.rng.geometric(1.0 / mean)
        self.cur[mask, f] = val
        self.left[mask, f] = dur

    def act(self, st):
        self.left -= 1
        for f in range(3):
            self._resample(f, self.left[:, f] <= 0)
        perp, _, _, _ = self.world.cast(st, self.PROBE)
        d_left = perp[:, :2].mean(1)
        d_ahead = perp[:, 2]
        d_right = perp[:, 3:].mean(1)
        close = (self.cur[:, 0] == rc.MOVE_FWD) & (d_ahead < self.avoid_dist)
        if close.any():
            turn_dir = np.where(d_left > d_right, rc.TURN_LEFT, rc.TURN_RIGHT)
            self.cur[close, 2] = turn_dir[close]
            self.left[close, 2] = self.rng.integers(4, 10, size=close.sum())
            stop = close & (self.rng.random(self.n) < 0.4)
            self.cur[stop, 0] = rc.MOVE_NONE
            self.left[stop, 0] = self.rng.integers(2, 6, size=stop.sum())
        return self.cur.copy()

    def random_states(self, n):
        """Uniform random poses over open space, respecting the collision margin next to walls."""
        w, R = self.world, self.world.spec["radius"]
        idx = self.rng.integers(0, len(self.free), n)
        c, r = self.free[idx, 0], self.free[idx, 1]
        g = w.grid
        wall_l = g[r, np.clip(c - 1, 0, 15)] > 0
        wall_r = g[r, np.clip(c + 1, 0, 15)] > 0
        wall_u = g[np.clip(r - 1, 0, 15), c] > 0
        wall_d = g[np.clip(r + 1, 0, 15), c] > 0
        x = c + R * wall_l + self.rng.random(n) * (1.0 - R * wall_l - R * wall_r)
        y = r + R * wall_u + self.rng.random(n) * (1.0 - R * wall_u - R * wall_d)
        ang = self.rng.random(n) * (2 * np.pi)
        return rc.state_from_pose(x, y, ang)

    def maybe_respawn(self, st):
        if self.respawn_prob <= 0:
            return st
        mask = self.rng.random(self.n) < self.respawn_prob
        if not mask.any():
            return st
        fresh = self.random_states(self.n)
        self.reset(mask)
        return rc.State(*[np.where(mask, f, s) for f, s in zip(fresh, st)])
