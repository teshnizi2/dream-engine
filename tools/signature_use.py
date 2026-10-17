#!/usr/bin/env python
"""Exploratory (docs/PREREG-v3.md deviation log 5): does a v3 model use the colour signature to localise itself?

The flat-wall escape diagnostic (eval_prereg.flat_escape) is run twice on the same poses: in the real v3 game, and in a copy of the
game whose tint is shifted by a constant phase per channel, which paints a different position code on the same geometry. In both runs
the context frames and the real reference frames come from the same game, so a model that only copies colours scores the same in both.
A model that reads the signature as a position (and has memorised the map) is told it is somewhere else and predicts the wrong
geometry: its PSNR drops. The drop measures how much the model relies on the signature.

    python tools/signature_use.py --ckpt runs/full2/final_ema.pt --name full2
"""
import argparse
import copy
import json
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import eval as E  # noqa: E402
import eval_prereg as P  # noqa: E402
from export import ckpt_game, load_ema  # noqa: E402
from game import raycaster as rc  # noqa: E402

PHASE_SHIFT = (0.31, 0.47, 0.59)


def shifted_spec(spec: dict) -> dict:
    """The same game with a different signature: every wave's phase moved by a constant (geometry, shading and fog untouched)."""
    s = copy.deepcopy(spec)
    for w, d in zip(s["tint"]["waves"], PHASE_SHIFT):
        w[2] = round((w[2] + d) % 1.0, 4)
    return s


def signature_use(denoise, K, n=64, horizon=24, steps=2, ctx_sigma=0.05, init="unit", spec=None):
    spec = spec or rc.load_spec(ROOT / "game" / "spec_v3.json")
    kw = dict(K=K, n=n, horizon=horizon, steps=steps, ctx_sigma=ctx_sigma, init=init)
    true = P.flat_escape(rc.World(spec), denoise, **kw)
    other = P.flat_escape(rc.World(shifted_spec(spec)), denoise, **kw)
    keep = ("episodes", "dream_psnr_at", "freeze_psnr_at", "dream_psnr_mean", "dream_flat_share_at_end", "real_flat_share_at_end")
    t, s = ({k: v[k] for k in keep} for v in (true, other))
    return {"true_signature": t, "shifted_signature": s,
            "psnr_drop_at_8": t["dream_psnr_at"]["8"] - s["dream_psnr_at"]["8"], "psnr_drop_at_16": t["dream_psnr_at"]["16"] - s["dream_psnr_at"]["16"],
            "psnr_drop_mean": t["dream_psnr_mean"] - s["dream_psnr_mean"],
            "definition": "flat-wall escape in the v3 game and in a copy with the tint phases shifted by %s; each run is scored against its own game's real frames" % (list(PHASE_SHIFT),)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--game", default=None, help="override the game recorded in the checkpoint (it must be a v3 spec)")
    a = ap.parse_args()
    dev = E.pick_device(a.device)
    model, step = load_ema(a.ckpt)
    model = model.to(dev).eval()
    game = a.game or ckpt_game(a.ckpt)
    assert "v3" in pathlib.Path(game).name, "the signature probe needs a model trained on the v3 game"
    out = {"script": "python tools/signature_use.py " + " ".join(sys.argv[1:]), "checkpoint": a.ckpt, "step": step, "num_ctx": model.cfg.num_ctx,
           **signature_use(model.denoise, model.cfg.num_ctx, n=a.n)}
    (ROOT / "results" / f"signature_use_{a.name}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k.startswith("psnr_drop")}), "true", {k: round(v, 1) for k, v in out["true_signature"]["dream_psnr_at"].items()},
          "shifted", {k: round(v, 1) for k, v in out["shifted_signature"]["dream_psnr_at"].items()})


if __name__ == "__main__":
    main()
