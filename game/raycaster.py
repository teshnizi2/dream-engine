"""Deterministic Wolfenstein-style raycaster, vectorised over a batch of independent games.

This is the reference implementation. `raycaster.mjs` is its JavaScript twin and must produce the
same frames (tests/parity.test.mjs). To make that possible without depending on libm:

* all state updates and shading use only + - * / floor abs sqrt on float64 (IEEE-exact in both
  languages), and the same expression order;
* the heading is stored as a unit vector (dx, dy) and rotated by a precomputed (cos, sin) pair, so
  there is no sin/cos/atan in the simulation loop;
* colours are quantised with floor(c + 0.5).

Coordinates: x grows east (map column), y grows south (map row); heading (1, 0) looks east, and a
positive rotation turns right. The camera plane is (-dy, dx) * plane, so screen-right = +plane.
"""
import json
import pathlib
from collections import namedtuple

import numpy as np

HERE = pathlib.Path(__file__).parent

State = namedtuple("State", ["x", "y", "dx", "dy"])   # each: float64 array of shape [B]

MOVE_NONE, MOVE_FWD, MOVE_BACK = 0, 1, 2
STRAFE_NONE, STRAFE_LEFT, STRAFE_RIGHT = 0, 1, 2
TURN_NONE, TURN_LEFT, TURN_RIGHT = 0, 1, 2


def load_spec(path=None):
    return json.loads(pathlib.Path(path or HERE / "spec.json").read_text())


def state_from_pose(x, y, angle):
    """Convenience constructor (uses trig; only for tests / data generation, never inside the sim)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    a = np.asarray(angle, dtype=np.float64)
    return State(x.copy(), y.copy(), np.cos(a), np.sin(a))


def state_from_spec_start(spec, batch=1):
    s = spec["start"]
    f = lambda v: np.full(batch, v, dtype=np.float64)
    return State(f(s["x"]), f(s["y"]), f(s["dx"]), f(s["dy"]))


def _frac(a):
    return a - np.floor(a)


def pattern(name, p, u, v):
    """Wall texture. u, v in [0, 1): position along the wall face / down the wall.
    Returns (k, m, em): colour-mix mask (0 -> rgb, 1 -> rgb2), brightness multiplier, emission mask."""
    one = np.ones_like(u)
    zero = np.zeros_like(u)
    if name == "blocks":
        rows, cols, mv, mu = p
        vr = v * rows
        rr = np.floor(vr)
        odd = rr - 2.0 * np.floor(rr * 0.5)
        uu = u * cols + 0.5 * odd
        k = ((vr - rr < mv) | (_frac(uu) < mu)).astype(np.float64)
        return k, one, zero
    if name == "frame":
        t = p[0]
        k = ((u < t) | (u >= 1.0 - t) | (v < t) | (v >= 1.0 - t)).astype(np.float64)
        return k, one, zero
    if name == "pillar":
        rings, ringw = p
        t = 2.0 * u - 1.0
        m = 0.6 + 0.4 * (1.0 - t * t)
        k = (_frac(v * rings) < ringw).astype(np.float64)
        return k, m, one
    if name == "windows":
        n, a0, a1, v0, v1 = p
        a = _frac(u * n)
        k = ((a >= a0) & (a < a1) & (v >= v0) & (v < v1)).astype(np.float64)
        return k, one, k
    if name == "moss":
        n = p[0]
        w = np.floor(u * n) * 7.0
        r = w - 5.0 * np.floor(w / 5.0)
        k = (r < 2.0).astype(np.float64)
        return k, 0.8 + 0.2 * v, zero
    if name == "dado":
        v0, lw = p
        k = (v >= v0).astype(np.float64)
        m = np.where((v >= v0) & (v < v0 + lw), 1.3, 1.0)
        return k, m, zero
    raise ValueError(name)


class World:
    def __init__(self, spec=None):
        spec = spec or load_spec()
        self.spec = spec
        self.N = spec["size"]
        self.W = spec["width"]
        self.H = spec["height"]
        chars = spec["wall_chars"]
        self.grid = np.array([[0 if ch == "." else chars[ch] for ch in row] for row in spec["map"]], dtype=np.int64)
        assert self.grid.shape == (self.N, self.N)
        walls = spec["walls"]
        nid = max(int(k) for k in walls) + 1
        self.rgb1 = np.zeros((nid, 3))
        self.rgb2 = np.zeros((nid, 3))
        self.emit = np.zeros(nid)
        self.patterns = {}
        for k, w in walls.items():
            i = int(k)
            self.rgb1[i] = w["rgb"]
            self.rgb2[i] = w["rgb2"]
            self.emit[i] = w["emit"]
            self.patterns[i] = (w["pattern"], w["p"])
        self.plane = spec["plane"]
        self.fog_k = spec["fog_k"]
        self.fog = np.array(spec["fog_color"], dtype=np.float64)
        self.side_shade = spec["side_shade"]
        self.camx = (2.0 * (np.arange(self.W) + 0.5)) / self.W - 1.0
        self.bg_rows = self._background_rows()

    # ------------------------------------------------------------------ background (ceiling / floor)
    def _background_rows(self):
        H = self.H
        half = H / 2.0
        out = np.zeros((H, 3), dtype=np.uint8)
        sp = self.spec
        for y in range(H):
            yy = (y + 0.5) - half
            s = abs(yy) / half
            rd = half / abs(yy)
            t = rd * self.fog_k
            f = 1.0 / (1.0 + t * t)
            if yy < 0:
                a, b = np.array(sp["ceiling"]["horizon"], dtype=np.float64), np.array(sp["ceiling"]["top"], dtype=np.float64)
            else:
                a, b = np.array(sp["floor"]["horizon"], dtype=np.float64), np.array(sp["floor"]["bottom"], dtype=np.float64)
            col = a + (b - a) * s
            col = self.fog + (col - self.fog) * f
            out[y] = np.clip(np.floor(col + 0.5), 0, 255).astype(np.uint8)
        return out

    # ------------------------------------------------------------------ geometry
    def free_cells(self):
        r, c = np.nonzero(self.grid == 0)
        return np.stack([c, r], 1)

    def _cell(self, cx, cy):
        return self.grid[np.clip(cy, 0, self.N - 1), np.clip(cx, 0, self.N - 1)]

    def cast(self, st, camx):
        """DDA for rays through screen positions `camx` [K]. Returns perp [B,K], side [B,K], wall id [B,K], u [B,K]."""
        x = st.x[:, None]
        y = st.y[:, None]
        px = (-st.dy * self.plane)[:, None]
        py = (st.dx * self.plane)[:, None]
        rdx = st.dx[:, None] + px * camx[None, :]
        rdy = st.dy[:, None] + py * camx[None, :]
        B, K = rdx.shape
        mapx = np.broadcast_to(np.floor(x).astype(np.int64), (B, K)).copy()
        mapy = np.broadcast_to(np.floor(y).astype(np.int64), (B, K)).copy()
        with np.errstate(divide="ignore", invalid="ignore"):
            ddx = np.abs(1.0 / rdx)
            ddy = np.abs(1.0 / rdy)
        stepx = np.where(rdx < 0, -1, 1)
        stepy = np.where(rdy < 0, -1, 1)
        with np.errstate(invalid="ignore"):
            sdx = np.where(rdx < 0, (x - mapx) * ddx, (mapx + 1.0 - x) * ddx)
            sdy = np.where(rdy < 0, (y - mapy) * ddy, (mapy + 1.0 - y) * ddy)
        hit = np.zeros((B, K), dtype=bool)
        side = np.zeros((B, K), dtype=np.int64)
        wid = np.zeros((B, K), dtype=np.int64)
        for _ in range(2 * self.N + 2):
            adv = ~hit
            xs = sdx < sdy
            mx = adv & xs
            my = adv & ~xs
            sdx = np.where(mx, sdx + ddx, sdx)
            mapx = np.where(mx, mapx + stepx, mapx)
            sdy = np.where(my, sdy + ddy, sdy)
            mapy = np.where(my, mapy + stepy, mapy)
            side = np.where(adv, np.where(xs, 0, 1), side)
            cell = self._cell(mapx, mapy)
            nh = adv & (cell > 0)
            wid = np.where(nh, cell, wid)
            hit = hit | nh
            if hit.all():
                break
        with np.errstate(invalid="ignore"):
            perp = np.where(side == 0, sdx - ddx, sdy - ddy)
        perp = np.maximum(perp, 1e-4)
        wx = np.where(side == 0, y + perp * rdy, x + perp * rdx)
        wx = wx - np.floor(wx)
        flip = ((side == 0) & (rdx > 0)) | ((side == 1) & (rdy < 0))
        u = np.where(flip, 1.0 - wx, wx)
        return perp, side, wid, u

    def distance_ahead(self, st):
        perp, _, _, _ = self.cast(st, np.array([0.0]))
        return perp[:, 0]

    # ------------------------------------------------------------------ rendering
    def render(self, st):
        H, W = self.H, self.W
        perp, side, wid, u = self.cast(st, self.camx)                      # [B, W]
        B = perp.shape[0]
        lineh = H / perp
        half = H / 2.0
        top = half - lineh * 0.5
        bot = half + lineh * 0.5
        yc = (np.arange(H) + 0.5)[None, :, None]
        is_wall = (yc >= top[:, None, :]) & (yc < bot[:, None, :])        # [B, H, W]
        out = np.empty((B, H, W, 3), dtype=np.uint8)
        out[:] = self.bg_rows[None, :, None, :]
        bi, yi, xi = np.nonzero(is_wall)
        if len(bi) == 0:
            return out
        wtp = wid[bi, xi]
        up = u[bi, xi]
        vp = (yi + 0.5 - top[bi, xi]) / lineh[bi, xi]
        k = np.zeros(len(bi))
        m = np.ones(len(bi))
        em = np.zeros(len(bi))
        for i, (name, p) in self.patterns.items():
            sel = wtp == i
            if sel.any():
                k[sel], m[sel], em[sel] = pattern(name, p, up[sel], vp[sel])
        base = self.rgb1[wtp] + (self.rgb2[wtp] - self.rgb1[wtp]) * k[:, None]
        c = base * m[:, None]
        shade = np.where(side[bi, xi] == 1, self.side_shade, 1.0)
        c = c * shade[:, None]
        t = perp * self.fog_k
        fw = (1.0 / (1.0 + t * t))[bi, xi]
        e = self.emit[wtp] * em
        f2 = fw + (1.0 - fw) * e
        col = self.fog + (c - self.fog) * f2[:, None]
        out[bi, yi, xi] = np.clip(np.floor(col + 0.5), 0, 255).astype(np.uint8)
        return out

    # ------------------------------------------------------------------ dynamics
    def step(self, st, action):
        """action: int array [B, 3] = (move, strafe, turn). Turn first, then move along the new heading."""
        sp = self.spec
        move, strafe, turn = action[:, 0], action[:, 1], action[:, 2]
        c, s = sp["turn_cos"], sp["turn_sin"]
        sgn = np.where(turn == TURN_RIGHT, 1.0, np.where(turn == TURN_LEFT, -1.0, 0.0))
        ss = s * sgn
        rdx = st.dx * c - st.dy * ss
        rdy = st.dx * ss + st.dy * c
        norm = np.sqrt(rdx * rdx + rdy * rdy)
        rot = turn != TURN_NONE
        ndx = np.where(rot, rdx / norm, st.dx)
        ndy = np.where(rot, rdy / norm, st.dy)

        vf = np.where(move == MOVE_FWD, 1.0, np.where(move == MOVE_BACK, -1.0, 0.0))
        vs = np.where(strafe == STRAFE_RIGHT, 1.0, np.where(strafe == STRAFE_LEFT, -1.0, 0.0))
        fwd = vf * sp["move_speed"]
        sd = vs * sp["strafe_speed"]
        mx = ndx * fwd + (-ndy) * sd
        my = ndy * fwd + ndx * sd

        R = sp["radius"]
        sx = np.where(mx > 0, 1.0, np.where(mx < 0, -1.0, 0.0))
        nx = st.x + mx
        okx = self._cell(np.floor(nx + sx * R).astype(np.int64), np.floor(st.y).astype(np.int64)) == 0
        x = np.where(okx, nx, st.x)
        sy = np.where(my > 0, 1.0, np.where(my < 0, -1.0, 0.0))
        ny = st.y + my
        oky = self._cell(np.floor(x).astype(np.int64), np.floor(ny + sy * R).astype(np.int64)) == 0
        y = np.where(oky, ny, st.y)
        return State(x, y, ndx, ndy)
