"""Where does a model's 1-step prediction error live? Squared error share by what the pixel shows (floor/ceiling or a wall
type) and on strong vertical edges, over 256 fixed validation windows, 1 Euler step, no context noise.
    python tools/error_attribution.py --ckpt runs/w2_B_final_ema.pt --name w2_B --out results/error_attribution_w2_B.json
Run it from a checkout of an older commit to attribute the error of a model trained on an older game version."""
import argparse
import json
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.loader import TrajectorySource  # noqa: E402
from export import load_ema  # noqa: E402
from game import raycaster as rc  # noqa: E402
from model import evaluate as ev  # noqa: E402
from model.denoiser import encode_actions  # noqa: E402
from model.sampler import build_sigmas, sample_next  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=256)
a = ap.parse_args()
torch.set_num_threads(4)
w = rc.World()
b = TrajectorySource(w, a.n, 5, seed=999).next_batch()
model, step = load_ema(a.ckpt)
fr = ev._to_unit(b["frames"])
ac = torch.from_numpy(b["actions"])
ctx, tgt = fr[:, :4], fr[:, 4]
with torch.no_grad():
    pred = sample_next(model.denoise, ctx, encode_actions(ac[:, :4]), torch.zeros(a.n), build_sigmas(1), quantize=True)
err = ((pred - tgt) / 2) ** 2
st = rc.State(*[b["poses"][:, 4, i] for i in range(4)])
perp, side, wid, u = w.cast(st, w.camx)
lineh = 64 / perp
top, bot = 32 - lineh * 0.5, 32 + lineh * 0.5
yc = (np.arange(64) + 0.5)[None, :, None]
is_wall = (yc >= top[:, None, :]) & (yc < bot[:, None, :])
wallid = np.where(is_wall, wid[:, None, :], 0)
e = err.mean(1).numpy()
names = {0: "floor_ceiling", **{int(k): v["name"] for k, v in w.spec["walls"].items()}}
tot = e.sum()
by = {}
for k, n in names.items():
    m = wallid == k
    if m.sum():
        by[n] = {"error_share_pct": float(100 * e[m].sum() / tot), "pixel_share_pct": float(100 * m.mean()), "mean_mse": float(e[m].mean())}
g = (tgt[..., 1:] - tgt[..., :-1]).abs().amax(1).numpy()
edge = np.zeros_like(e, dtype=bool)
edge[..., 1:] |= g > 0.15
edge[..., :-1] |= g > 0.15
res = {"script": "python tools/error_attribution.py " + " ".join(sys.argv[1:]), "checkpoint": a.ckpt, "step": step, "windows": a.n,
       "pooled_psnr": float(10 * np.log10(1 / e.mean())), "by_surface": by,
       "strong_vertical_edges": {"pixel_share_pct": float(100 * edge.mean()), "error_share_pct": float(100 * e[edge].sum() / tot)}}
pathlib.Path(a.out).write_text(json.dumps(res, indent=1))
print(json.dumps({k: res[k] for k in ("pooled_psnr", "strong_vertical_edges")}), {n: round(v["error_share_pct"], 1) for n, v in by.items()})
