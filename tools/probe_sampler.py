#!/usr/bin/env python
"""Post-hoc inference-setting probe: for one checkpoint, roll out 128 frames in pure imagination for a grid of
sampler settings (Euler steps 1/2/3, init sigma_max*N vs N(0,1), context-noise level) and report drift (PSNR vs the
real game), sharpness relative to the real frames, and PSNR along the scripted tour used in the video.

    python tools/probe_sampler.py --ckpt runs/full/final_ema.pt --name full --device cuda
Writes results/probe_<name>.json."""
import argparse
import itertools
import json
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from game import raycaster as rc  # noqa: E402
from model import evaluate as ev  # noqa: E402
from export import ckpt_game, load_ema  # noqa: E402


def grad_mag(x):                                   # x [..., 3, H, W] in [-1,1]
    gx = (x[..., :, 1:] - x[..., :, :-1]).abs().mean()
    gy = (x[..., 1:, :] - x[..., :-1, :]).abs().mean()
    return float((gx + gy) / 2)


def tour_inputs(world, n_ticks, K=4):
    tour = json.loads((ROOT / "site/js/tour.json").read_text())["actions"][:n_ticks]
    st = rc.state_from_spec_start(world.spec)
    frames = [world.render(st)[0]]
    acts = []
    for a in [[0, 0, 0]] * (K - 1) + tour:
        acts.append(a)
        st = world.step(st, np.array([a], dtype=np.int64))
        frames.append(world.render(st)[0])
    frames = ev._to_unit(np.stack(frames))                      # [T+K, 3, 64, 64]
    return {"frames": frames[None], "actions": torch.tensor(acts, dtype=torch.long)[None]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--horizon", type=int, default=128)
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--tour", type=int, default=480)
    a = ap.parse_args()
    dev = torch.device(a.device)
    model, step = load_ema(a.ckpt)
    model = model.to(dev).eval()
    K = model.cfg.num_ctx
    world = rc.World(rc.load_spec(ckpt_game(a.ckpt)))
    H = a.horizon
    traj = ev.eval_trajectories_t(world, n=a.episodes, length=K + H, seed=321)
    gt_sharp = grad_mag(traj["frames"][:, K:K + H])
    out = {"script": "python tools/probe_sampler.py " + " ".join(sys.argv[1:]), "checkpoint": a.ckpt, "step": step,
           "horizon": H, "episodes": a.episodes, "gt_gradient_magnitude": gt_sharp, "grid": []}
    tour = tour_inputs(world, a.tour, K)
    for steps, init, cs in itertools.product((1, 2, 3), ("edm", "unit"), (0.0, 0.1)):
        r = ev.rollout_metrics(model.denoise, traj, horizon=H, steps=steps, order=1, ctx_sigma=cs, init=init, return_frames=True, batch=16, K=K)
        sharp = grad_mag(r["frames"][:, -32:]) / grad_mag(traj["frames"][:, K + H - 32:K + H])
        rt = ev.rollout_metrics(model.denoise, tour, horizon=a.tour, steps=steps, order=1, ctx_sigma=cs, init=init, batch=1, K=K)
        row = {"steps": steps, "init": init, "ctx_sigma": cs,
               "psnr_at": {str(k): float(r["psnr"][k - 1]) for k in (1, 4, 8, 16, 32, 64, 128) if k <= H},
               "mean_psnr_32": float(r["psnr"][:32].mean()), "mean_psnr_128": float(r["psnr"][:H].mean()),
               "ssim_at_32": float(r["ssim"][31]), "sharpness_ratio_last32": sharp,
               "tour_psnr_at": {str(k): float(rt["psnr"][k - 1]) for k in (1, 16, 32, 64, 128, 240, 480) if k <= a.tour},
               "tour_mean_psnr": float(rt["psnr"].mean())}
        out["grid"].append(row)
        print(f"steps={steps} init={init:4s} ctx={cs}: mean32 {row['mean_psnr_32']:.2f} mean128 {row['mean_psnr_128']:.2f} "
              f"sharp {sharp:.2f} tour_mean {row['tour_mean_psnr']:.2f}", flush=True)
    res = ROOT / "results"
    res.mkdir(exist_ok=True)
    (res / f"probe_{a.name}.json").write_text(json.dumps(out, indent=1))
    # visual comparison: truth and a few sampler settings on two episodes, frames 1 8 16 32 64 128 (3x upscaled)
    from PIL import Image
    cols = [k for k in (1, 8, 16, 32, 64, 128) if k <= H]
    rows = []
    two = {k: v[:2] for k, v in traj.items()}
    rows.append(("truth", torch.cat([torch.cat([two["frames"][e, K - 1 + k] for k in cols], 2) for e in range(2)], 1)))
    for steps, init, cs in ((1, "unit", 0.05), (2, "unit", 0.05), (3, "unit", 0.05), (1, "edm", 0.05), (3, "edm", 0.05)):
        r = ev.rollout_metrics(model.denoise, two, horizon=H, steps=steps, order=1, ctx_sigma=cs, init=init, return_frames=True, batch=2, K=K)
        rows.append((f"{steps}-step {init} ctx{cs}", torch.cat([torch.cat([r["frames"][e, k - 1] for k in cols], 2) for e in range(2)], 1)))
    pad = torch.full((3, 4, rows[0][1].shape[2]), -0.8)
    img = torch.cat([x for _, x in rows for x in (x, pad)], 1)
    arr = ((img + 1) / 2 * 255).round().clamp(0, 255).byte().permute(1, 2, 0).numpy()
    Image.fromarray(arr).resize((arr.shape[1] * 3, arr.shape[0] * 3), Image.NEAREST).save(res / f"probe_{a.name}_grid.png")
    (res / f"probe_{a.name}_grid.txt").write_text("rows: " + ", ".join(n for n, _ in rows) + "; columns = frames " + str(cols))


if __name__ == "__main__":
    main()
