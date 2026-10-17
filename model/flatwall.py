"""Flat-wall metric (pre-registered in docs/PREREG-v3.md).

A *flat-wall frame* is a frame in which (almost) no floor or ceiling is visible: the wall fills the view. This is a
statement about geometry, not about texture, so it means the same on a real frame, a dreamed frame and a frame of a
different version of the game (the floor and ceiling are painted by row and never change between game versions).

* `background_fraction`: the share of pixels whose colour is the floor/ceiling colour of their row (the game's own
  `bg_rows` table), within BG_TOL grey levels because a dreamed frame reproduces it only approximately.
* `flat_flags`: background fraction <= FLAT_MAX_BG.
* `stuck_flags`: the dream is *stuck* on a flat wall at frame t if the last `window` frames (t-window+1 .. t) are all flat.
  A single wall close-up is ordinary (the real game spends about a quarter of its frames facing a wall); what the
  dream does wrong is stay there.
* `collapse_rate`: over frames t in [64, 512] since wake-up, the fraction of dream frames that are stuck on a flat wall
  while the matching real frame is not.
"""
import numpy as np

BG_TOL = 4            # grey levels (max over the three channels) a pixel may differ from its row's floor/ceiling colour
FLAT_MAX_BG = 0.05    # a frame is flat-wall if at most 5% of its pixels are floor or ceiling
STUCK_WINDOW = 32     # frames in a row that must all be flat for the dream to count as stuck
T_RANGE = (64, 512)   # frames since wake-up (1 = the first dreamed frame), both ends included


def background_fraction(frames_u8: np.ndarray, bg_rows: np.ndarray, tol: int = BG_TOL) -> np.ndarray:
    """frames uint8 [..., H, W, 3]; bg_rows uint8 [H, 3] -> float [...]: share of pixels within `tol` of their row's colour."""
    f = np.asarray(frames_u8).astype(np.int16)
    d = np.abs(f - np.asarray(bg_rows).astype(np.int16)[:, None, :]).max(-1)       # [..., H, W]
    return (d <= tol).mean((-1, -2))


def flat_flags(bg_frac: np.ndarray, max_bg: float = FLAT_MAX_BG) -> np.ndarray:
    return np.asarray(bg_frac) <= max_bg


def stuck_flags(flat: np.ndarray, window: int = STUCK_WINDOW) -> np.ndarray:
    """flat bool [E, T] -> bool [E, T]: True where the last `window` frames (this one included) are all flat. The first
    window-1 frames can never be stuck: nothing before the first dreamed frame counts."""
    flat = np.asarray(flat, dtype=bool)
    c = np.cumsum(np.pad(flat, ((0, 0), (1, 0))).astype(np.int64), axis=1)         # c[:, j] = flat frames among the first j
    T = flat.shape[1]
    out = np.zeros_like(flat)
    if window <= T:
        out[:, window - 1:] = (c[:, window:] - c[:, :T - window + 1]) == window
    return out


def collapse_rate(dream_flat: np.ndarray, real_flat: np.ndarray, window: int = STUCK_WINDOW, t_range=T_RANGE) -> dict:
    """Fraction of (episode, frame t) with t in `t_range` where the dream is stuck on a flat wall and the real game is not
    (the real game's stuck flag is computed with the same window). Frame numbers count from 1."""
    lo, hi = t_range
    d = stuck_flags(dream_flat, window)[:, lo - 1:hi]
    r = stuck_flags(real_flat, window)[:, lo - 1:hi]
    if d.shape[1] == 0:                       # a rollout that ends before frame `lo`: nothing to measure
        nan = np.full(d.shape[0], np.nan)
        return {"rate": float("nan"), "per_episode": nan, "dream_stuck_share": float("nan"), "real_stuck_share": float("nan"),
                "window": int(window), "t_range": [int(lo), int(hi)]}
    per_ep = (d & ~r).mean(1)
    return {"rate": float(per_ep.mean()), "per_episode": per_ep, "dream_stuck_share": float(d.mean()),
            "real_stuck_share": float(r.mean()), "window": int(window), "t_range": [int(lo), int(hi)]}


def bootstrap_ci(values: np.ndarray, n: int = 4000, seed: int = 0, stat=np.mean, alpha: float = 0.05):
    """Percentile bootstrap interval of `stat` over episodes (resampling whole episodes)."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), (n, len(v)))
    s = np.array([stat(v[i]) for i in idx])
    return float(np.quantile(s, alpha / 2)), float(np.quantile(s, 1 - alpha / 2))


def run_lengths(flat: np.ndarray) -> list:
    """Lengths of all maximal runs of True in each row of a bool [E, T] array."""
    out = []
    for row in np.asarray(flat, dtype=bool):
        cur = 0
        for v in row:
            if v:
                cur += 1
            elif cur:
                out.append(cur)
                cur = 0
        if cur:
            out.append(cur)
    return out
