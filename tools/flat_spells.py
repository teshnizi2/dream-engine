#!/usr/bin/env python
"""Exploratory: how long do flat-wall spells last in a dream, against the real game? From the per-frame arrays saved by eval_prereg.py
(results/prereg_<run>.npz). Not part of any pre-registered decision; it describes *how* the collapse differs between runs
(longer spells, a smaller chance of leaving a flat wall), which the collapse rate alone does not.

    python tools/flat_spells.py w2_B v2k8 v3k4 v3k8 full1        # writes results/flat_spells.json
"""
import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import flatwall as fw  # noqa: E402


def spells(bg: np.ndarray, t_from: int = 64) -> dict:
    """Flat-wall spells (maximal runs of flat frames) among frames t >= t_from of bg [episodes, frames] (background fractions)."""
    flat = fw.flat_flags(np.asarray(bg)[:, t_from - 1:])
    runs, ended = [], []
    for row in flat:
        cur = 0
        for i, v in enumerate(row):
            if v:
                cur += 1
            elif cur:
                runs.append(cur)
                ended.append(cur)
                cur = 0
        if cur:
            runs.append(cur)
    has_next = flat[:, :-1]
    leaves = has_next & ~flat[:, 1:]
    n_flat_with_successor = int(has_next.sum())
    return {"n_spells": len(runs), "n_ended": len(ended), "longest": int(max(runs)) if runs else 0,
            "median_length": float(np.median(runs)) if runs else 0.0,
            "median_length_of_ended": float(np.median(ended)) if ended else 0.0,
            "share_of_frames_flat": float(flat.mean()),
            "chance_of_leaving_per_flat_frame": float(leaves.sum() / n_flat_with_successor) if n_flat_with_successor else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--t-from", type=int, default=64)
    a = ap.parse_args()
    out = {"script": "python tools/flat_spells.py " + " ".join(sys.argv[1:]), "frames_from": a.t_from, "runs": {}}
    real = None
    for r in a.runs:
        z = np.load(ROOT / "results" / f"prereg_{r}.npz")
        out["runs"][r] = {"dream": spells(z["bg_dream"], a.t_from)}
        real = real if real is not None else {"real_game": spells(z["bg_real"], a.t_from)}
    out["real_game"] = real["real_game"]
    (ROOT / "results" / "flat_spells.json").write_text(json.dumps(out, indent=1))
    print(f"{'run':10s} {'share flat':>10s} {'spells':>7s} {'median len':>10s} {'longest':>8s} {'P(leave)/frame':>14s}")
    for r, v in [("real game", {"dream": out["real_game"]})] + list(out["runs"].items()):
        d = v["dream"]
        print(f"{r:10s} {d['share_of_frames_flat']:10.3f} {d['n_spells']:7d} {d['median_length']:10.1f} {d['longest']:8d} {d['chance_of_leaving_per_flat_frame']:14.4f}")


if __name__ == "__main__":
    main()
