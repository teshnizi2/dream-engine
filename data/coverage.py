"""Coverage heatmap of the training distribution: where on the map (and facing which way) do the
training windows sit?  Writes results/coverage.json and results/coverage.png.

    python -m data.coverage
"""
import json
import pathlib

import numpy as np

from game import raycaster as rc
from data.loader import TrajectorySource

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main(n_batches=400, batch=64, seed=0):
    w = rc.World()
    src = TrajectorySource(w, batch, 5, seed=seed)
    cells = np.zeros((16, 16))
    heading = np.zeros((16, 16, 8))
    for _ in range(n_batches):
        b = src.next_batch()
        p = b["poses"][:, -1]                                       # pose at the target frame
        cx, cy = np.floor(p[:, 0]).astype(int), np.floor(p[:, 1]).astype(int)
        np.add.at(cells, (cy, cx), 1)
        ang = np.arctan2(p[:, 3], p[:, 2])
        k = np.floor((ang + np.pi) / (2 * np.pi) * 8).astype(int) % 8
        np.add.at(heading, (cy, cx, k), 1)
    free = w.grid == 0
    c = cells[free]
    prob = c / c.sum()
    uniform = 1.0 / free.sum()
    stats = {
        "script": "python -m data.coverage",
        "samples": int(cells.sum()),
        "open_cells": int(free.sum()),
        "cells_with_zero_samples": int((c == 0).sum()),
        "min_cell_share_vs_uniform": float(prob.min() / uniform),
        "max_cell_share_vs_uniform": float(prob.max() / uniform),
        "normalised_entropy": float(-(prob * np.log(prob)).sum() / np.log(len(prob))),
        "heading_bins_empty_fraction": float((heading[free] == 0).mean()),
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "coverage.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    spec = w.spec
    fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=160)
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    heat = np.where(free, cells / cells[free].mean(), np.nan)
    im = ax.imshow(heat, cmap="magma", vmin=0, vmax=2.5, interpolation="nearest")
    for r in range(16):
        for cc in range(16):
            if not free[r, cc] and spec["map"][r][cc] != "S":
                col = np.array(spec["walls"][str(spec["wall_chars"][spec["map"][r][cc]])]["rgb"]) / 255.0
                ax.add_patch(plt.Rectangle((cc - 0.5, r - 0.5), 1, 1, color=col, alpha=0.9, lw=0))
            elif not free[r, cc]:
                ax.add_patch(plt.Rectangle((cc - 0.5, r - 0.5), 1, 1, color="#2e3a52", alpha=0.7, lw=0))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Training-sample density per cell (1.0 = mean)", color="#e8ecf5", fontsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.ax.tick_params(colors="#8893ab", labelsize=7)
    fig.savefig(ROOT / "results" / "coverage.png", facecolor=fig.get_facecolor(), bbox_inches="tight")
    print("wrote results/coverage.png")


if __name__ == "__main__":
    main()
