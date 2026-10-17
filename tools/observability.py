#!/usr/bin/env python
"""Can a flat wall view tell where the player is? A model-free test of the observability explanation of the flat-wall collapse.

For one game version: render a bank of frames at uniformly random poses (position and heading known) and a set of query frames at
other random poses, keep the queries that are flat wall (model/flatwall.py: the wall fills the view), add Gaussian noise to each
query (a dreamed frame is never exact), and recover the pose of each query as the pose of its nearest neighbour in the bank (colours
pooled to 8 x 8 blocks). Reports how far off that recovered pose is: the information a flat wall view carries about where you are,
independent of any network. The same procedure on the v2 and v3 games shows what the signature layer adds.

    python tools/observability.py          # writes results/observability_v2_v3.json
"""
import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.policy import ExplorationPolicy  # noqa: E402
from game import raycaster as rc  # noqa: E402
from model import flatwall as fw  # noqa: E402


def _poses(world, n, seed):
    st = ExplorationPolicy(world, n, np.random.default_rng(seed)).random_states(n)
    return np.stack(st, 1)


def _render(world, poses, chunk=512):
    return np.concatenate([world.render(rc.State(*poses[i:i + chunk].T)) for i in range(0, len(poses), chunk)])


def _features(frames_u8, pool=8):
    f = frames_u8.astype(np.float32)
    n = len(f)
    return f.reshape(n, 64 // pool, pool, 64 // pool, pool, 3).mean((2, 4)).reshape(n, -1)


def observability(world, n_bank=20000, n_query=6000, noise=4.0, seed=0, max_queries=1500):
    bank_p, q_p = _poses(world, n_bank, 1), _poses(world, n_query, 2)
    bank, q = _render(world, bank_p), _render(world, q_p)
    flat = fw.flat_flags(fw.background_fraction(q, world.bg_rows))
    q_p, q = q_p[flat][:max_queries], q[flat][:max_queries]
    rng = np.random.default_rng(seed)
    qn = np.clip(q.astype(np.float32) + rng.normal(0, noise, q.shape), 0, 255)
    fb, fq = _features(bank), _features(qn)
    d2 = (fq ** 2).sum(1, keepdims=True) + (fb ** 2).sum(1)[None] - 2 * fq @ fb.T
    nn = d2.argmin(1)
    dpos = np.hypot(bank_p[nn, 0] - q_p[:, 0], bank_p[nn, 1] - q_p[:, 1])
    dang = np.abs((np.arctan2(q_p[:, 3], q_p[:, 2]) - np.arctan2(bank_p[nn, 3], bank_p[nn, 2]) + np.pi) % (2 * np.pi) - np.pi)
    return {"n_flat_queries": int(len(q_p)), "flat_share_of_random_poses": float(flat.mean()), "noise_grey_levels": noise,
            "pos_err_median": float(np.median(dpos)), "pos_err_p90": float(np.percentile(dpos, 90)),
            "within_0.75_cells": float(np.mean(dpos < 0.75)), "within_1.5_cells": float(np.mean(dpos < 1.5)),
            "heading_err_median_deg": float(np.degrees(np.median(dang))), "heading_within_45deg": float(np.mean(dang < np.radians(45)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-bank", type=int, default=20000)
    ap.add_argument("--n-query", type=int, default=6000)
    ap.add_argument("--noise", type=float, nargs="+", default=[4.0, 8.0])
    a = ap.parse_args()
    out = {"script": "python tools/observability.py " + " ".join(sys.argv[1:]), "bank_frames": a.n_bank,
           "method": "nearest neighbour (8x8 pooled colours) among bank frames of uniformly random poses; queries are flat-wall frames plus Gaussian noise"}
    for name in ("v2", "v3"):
        world = rc.World(rc.load_spec(ROOT / "game" / f"spec_{name}.json"))
        out[name] = {f"noise_{n:g}": observability(world, a.n_bank, a.n_query, n) for n in a.noise}
        print(name, json.dumps(out[name]), flush=True)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "observability_v2_v3.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
