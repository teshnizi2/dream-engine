#!/usr/bin/env python
"""media/collapse_curve.png: the collapse rate against training steps for v2 (`full1`) and v3 (`full2`), from the pre-registered evaluation of
their milestone checkpoints (results/prereg_<run>_s<step>.json) and of the final models (results/prereg_<run>.json).

    python tools/render_collapse_curve.py
"""
import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

BG, PANEL, GRID, TEXT, MUTED, AMBER, BLUE = "#0e1117", "#151a23", "#2e3a52", "#e8ecf5", "#8893ab", "#f59e0b", "#7aa2f7"


def curves(res) -> dict:
    """{run: [(step, collapse rate)]} sorted by step, from the milestone files and the final file of full1 and full2."""
    res = pathlib.Path(res)
    out = {}
    for run in ("full1", "full2"):
        pts = {}
        for f in sorted(res.glob(f"prereg_{run}_s*.json")):
            pts[int(f.stem.rsplit("_s", 1)[1])] = json.loads(f.read_text())["collapse"]["rate"]
        fin = res / f"prereg_{run}.json"
        if fin.exists():
            j = json.loads(fin.read_text())
            if j.get("step"):
                pts[int(j["step"])] = j["collapse"]["rate"]
        if pts:
            out[run] = sorted(pts.items())
    return out


def draw(cv: dict, path, target=None, real_floor=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=130, facecolor=BG)
    ax.set_facecolor(PANEL)
    style = {"full1": ("v2 game, K = 4 (shipped)", BLUE), "full2": ("v3 game, K = 4 (signatures)", AMBER)}
    for run, pts in cv.items():
        label, col = style[run]
        ax.plot([s / 1000 for s, _ in pts], [v for _, v in pts], "-o", color=col, lw=2, ms=5, label=label)
    if target is not None:
        ax.axhline(target, color=AMBER, lw=1, ls="--", alpha=0.7)
        ax.text(21, target + 0.01, "adoption target: half of v2's final rate", color=AMBER, ha="left", va="bottom", fontsize=8)
    if real_floor is not None:
        ax.axhline(real_floor, color=MUTED, lw=1, ls=":")
        ax.text(150, real_floor - 0.008, "floor: a dream with the real game's own flat spells but no tracking", color=MUTED, ha="right", va="top", fontsize=8)
    ax.set_xlabel("training steps (thousands)", color=MUTED)
    ax.set_ylabel("collapse rate (stuck on a flat wall)", color=MUTED)
    ax.set_title("Flat-wall collapse against training steps (64 episodes x 512 frames, frames 64 to 512)", color=TEXT, fontsize=10, loc="left")
    ax.tick_params(colors=MUTED)
    ax.grid(color=GRID, lw=0.6, alpha=0.7)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.set_ylim(bottom=0)
    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "media" / "collapse_curve.png"))
    a = ap.parse_args()
    res = ROOT / "results"
    cv = curves(res)
    fin = res / "prereg_full1.json"
    target = json.loads(fin.read_text())["collapse"]["rate"] / 2 if fin.exists() else None
    real = None
    if fin.exists():
        c = json.loads(fin.read_text())["collapse"]["real_stuck_share"]
        real = c * (1 - c)
    pathlib.Path(a.out).parent.mkdir(exist_ok=True)
    draw(cv, a.out, target, real)
    print("wrote", a.out, {k: len(v) for k, v in cv.items()})


if __name__ == "__main__":
    main()
