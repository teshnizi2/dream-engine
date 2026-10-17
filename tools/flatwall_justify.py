#!/usr/bin/env python
"""Why the flat-wall constants of docs/PREREG-v3.md are what they are: tables computed from saved v2 rollouts.

Reads the per-frame arrays written by eval_prereg.py (`results/prereg_<name>.npz`: PSNR, background fraction of the dreamed and
of the matching real frames) and, optionally, the saved frames of a few episodes (`results/rollouts_prereg_<name>.npz`) for
the colour-tolerance check and a contact sheet. Writes results/flatwall_justification_<name>.json (+ .png).

    python tools/flatwall_justify.py --name full1_m4_32ep
"""
import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import flatwall as fw  # noqa: E402

MAX_BG = (0.0, 0.02, 0.05, 0.10, 0.15, 0.20)
WINDOWS = (1, 4, 8, 16, 24, 32, 48, 64, 96, 128)
BINS = ((1, 32), (33, 64), (65, 128), (129, 256), (257, 512))


def _pct(runs):
    if not runs:
        return {"n": 0}
    return {"n": len(runs), "p50": float(np.percentile(runs, 50)), "p75": float(np.percentile(runs, 75)),
            "p90": float(np.percentile(runs, 90)), "p95": float(np.percentile(runs, 95)), "max": int(max(runs))}


def justify(z) -> dict:
    bd, br = np.asarray(z["bg_dream"]), np.asarray(z["bg_real"])
    lo, hi = fw.T_RANGE
    flat_d, flat_r = fw.flat_flags(bd), fw.flat_flags(br)
    out = {"episodes": int(bd.shape[0]), "frames": int(bd.shape[1]), "t_range": [lo, hi]}
    out["flat_share_by_max_bg"] = {str(m): {"dream": float((bd[:, lo - 1:hi] <= m).mean()), "real": float((br[:, lo - 1:hi] <= m).mean())}
                                   for m in MAX_BG}
    out["flat_share_by_time"] = {f"{a}-{b}": {"dream": float(flat_d[:, a - 1:b].mean()), "real": float(flat_r[:, a - 1:b].mean())}
                                 for a, b in BINS if a <= bd.shape[1]}
    out["run_lengths"] = {"dream": _pct(fw.run_lengths(flat_d)), "real": _pct(fw.run_lengths(flat_r))}
    rows = []
    for w in WINDOWS:
        c = fw.collapse_rate(flat_d, flat_r, w)
        ci = fw.bootstrap_ci(c["per_episode"])
        floor = c["real_stuck_share"] * (1 - c["real_stuck_share"])
        rows.append({"window": w, "dream_stuck_share": c["dream_stuck_share"], "real_stuck_share": c["real_stuck_share"],
                     "collapse_rate": c["rate"], "collapse_rate_ci95": list(ci),
                     "floor_if_dream_had_the_real_rate_but_no_tracking": floor})
    out["window_table"] = rows
    out["chosen"] = {"bg_tol": fw.BG_TOL, "max_bg": fw.FLAT_MAX_BG, "window": fw.STUCK_WINDOW, "t_range": list(fw.T_RANGE)}
    d, r = np.asarray(bd[:, lo - 1:hi]).ravel(), np.asarray(br[:, lo - 1:hi]).ravel()
    out["background_fraction_deciles"] = {"dream": np.percentile(d, np.arange(10, 100, 10)).round(4).tolist(),
                                          "real": np.percentile(r, np.arange(10, 100, 10)).round(4).tolist()}
    return out


def tolerance_table(frames_u8, bg_rows, tols=(0, 1, 2, 4, 6, 10)) -> dict:
    """How the verdict depends on the colour tolerance: share of flat frames and mean background fraction, per tolerance."""
    res = {}
    for t in tols:
        bg = fw.background_fraction(frames_u8, bg_rows, tol=t)
        res[str(t)] = {"flat": float(fw.flat_flags(bg).mean()), "mean_bg": float(bg.mean())}
    return res


def contact_sheet(path, dream, gt, bd, br, times=(64, 96, 128, 192, 256, 384, 512), episodes=8, scale=2):
    """Real above dream, same frame number; green outline = not stuck on a flat wall, red = stuck (window of the metric)."""
    from PIL import Image, ImageDraw
    n = min(episodes, len(dream))
    sd = fw.stuck_flags(fw.flat_flags(bd[:n]), fw.STUCK_WINDOW)
    sr = fw.stuck_flags(fw.flat_flags(br[:n]), fw.STUCK_WINDOW)
    tile = 64 * scale
    img = Image.new("RGB", (len(times) * (tile + 4) + 44, n * (2 * tile + 12)), (14, 17, 23))
    dr = ImageDraw.Draw(img)
    for e in range(n):
        y0 = e * (2 * tile + 12)
        dr.text((2, y0 + 20), "REAL", fill=(180, 180, 180))
        dr.text((2, y0 + tile + 20), "DREAM", fill=(245, 158, 11))
        for j, t in enumerate(times):
            for r, (src, st) in enumerate(((gt, sr), (dream, sd))):
                x0, y = 44 + j * (tile + 4), y0 + r * tile
                img.paste(Image.fromarray(src[e, t - 1]).resize((tile, tile), Image.NEAREST), (x0, y))
                dr.rectangle([x0, y, x0 + tile - 1, y + tile - 1], outline=(245, 60, 60) if st[e, t - 1] else (60, 200, 90), width=3)
    img.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="reads results/prereg_<name>.npz and results/rollouts_prereg_<name>.npz")
    a = ap.parse_args()
    res = ROOT / "results"
    z = np.load(res / f"prereg_{a.name}.npz")
    out = {"script": "python tools/flatwall_justify.py " + " ".join(sys.argv[1:]), "source": f"results/prereg_{a.name}.npz",
           **justify(z)}
    fr = res / f"rollouts_prereg_{a.name}.npz"
    if fr.exists():
        from game import raycaster as rc
        f = np.load(fr)
        w = rc.World(rc.load_spec("game/spec_v2.json"))
        out["tolerance_table_dream_frames_t64plus"] = tolerance_table(f["dream"][:, 63:], w.bg_rows)
        out["tolerance_table_real_frames_t64plus"] = tolerance_table(f["gt"][:, 63:], w.bg_rows)
        contact_sheet(res / f"flatwall_sheet_{a.name}.png", f["dream"], f["gt"], z["bg_dream"], z["bg_real"])
    (res / f"flatwall_justification_{a.name}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("flat_share_by_max_bg", "run_lengths")}, indent=1))
    for r in out["window_table"]:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "collapse_rate_ci95"})


if __name__ == "__main__":
    main()
