#!/usr/bin/env python
"""How flat-wall is the dream in the video? Reads results/video_capture_psnr.json (written by tools/capture_video.mjs: the floor/ceiling
share of the dreamed and the real frame at every tick) and applies the pre-registered definitions of model/flatwall.py: flat-wall
share of the ticks, and stuck on a flat wall (the last 32 ticks all flat). Writes results/video_flat_stats.json.

    python tools/video_flat_stats.py [--in results/video_capture_psnr.json]
"""
import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import flatwall as fw  # noqa: E402


def stats(j: dict) -> dict:
    d = fw.flat_flags(np.array([j["bg_dream_per_tick"]], dtype=float))
    r = fw.flat_flags(np.array([j["bg_real_per_tick"]], dtype=float))
    sd, sr = fw.stuck_flags(d), fw.stuck_flags(r)
    return {"ticks": int(d.shape[1]), "wake_every": j.get("wake_every", 0),
            "dream_flat_share": float(d.mean()), "real_flat_share": float(r.mean()),
            "dream_stuck_share": float(sd.mean()), "real_stuck_share": float(sr.mean()),
            "dream_stuck_ticks_while_real_is_not": int((sd & ~sr).sum()),
            "window": fw.STUCK_WINDOW, "note": "one scripted tour, one run: an illustration, not an evaluation"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", default=str(ROOT / "results" / "video_capture_psnr.json"))
    ap.add_argument("--out", default=str(ROOT / "results" / "video_flat_stats.json"))
    a = ap.parse_args()
    j = json.loads(pathlib.Path(a.src).read_text())
    out = {"script": "python tools/video_flat_stats.py " + " ".join(sys.argv[1:]), "source": a.src, **stats(j)}
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
