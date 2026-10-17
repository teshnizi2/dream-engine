"""media/sampler_steps.png: the same K real context frames + actions, the next frame predicted with 1, 2 and 3 Euler steps
(demo sampler: `unit` start, context noise 0.05), next to the true next frame. Windows are evenly spaced picks from the
256 fixed validation windows of eval.py (not chosen by quality). Per-window PSNR goes to results/sampler_figure.json.

    python tools/sampler_figure.py --ckpt runs/full1_final_ema.pt --device mps
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from export import ckpt_game, load_ema  # noqa: E402
from game import raycaster as rc  # noqa: E402
from model import evaluate as ev  # noqa: E402
from model import metrics  # noqa: E402


def to_pil(x, size):
    from PIL import Image
    a = ((x + 1) / 2 * 255).round().clamp(0, 255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(a).resize((size, size), Image.NEAREST)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--ctx-sigma", type=float, default=0.05)
    ap.add_argument("--init", default="unit")
    ap.add_argument("--out", default=str(ROOT / "media/sampler_steps.png"))
    ap.add_argument("--json", default=str(ROOT / "results/sampler_figure.json"))
    a = ap.parse_args()
    dev = torch.device(a.device)
    model, step = load_ema(a.ckpt)
    model = model.to(dev).eval()
    K = model.cfg.num_ctx
    val = ev.val_windows(rc.World(rc.load_spec(ckpt_game(a.ckpt))), n=256, seed=999, K=K)
    idx = np.linspace(0, 255, a.n).astype(int)
    win = {k: v[idx].to(dev) for k, v in val.items()}
    preds, psnr = {}, {}
    for steps in (1, 2, 3):
        torch.manual_seed(0)
        r = ev.one_step_metrics(model.denoise, win, steps=steps, ctx_sigma=a.ctx_sigma, init=a.init, return_pred=True)
        preds[steps], psnr[steps] = r["pred"], r["psnr_per_window"]
    ctx_last = win["frames"][:, K - 1].cpu()
    target = win["frames"][:, K].cpu()

    from PIL import Image, ImageDraw, ImageFont
    tile, gap, left, top, pad = 192, 8, 24, 62, 22
    cols = ["last context frame", "true next frame", "1 Euler step", "2 steps (demo)", "3 steps"]
    bg, line, text, muted, amber = (14, 17, 23), (46, 58, 82), (232, 236, 245), (136, 147, 171), (245, 158, 11)
    W = left + len(cols) * (tile + gap) - gap + pad
    H = top + a.n * (tile + gap + 24) + pad
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)
    f_h, f_s = ImageFont.load_default(size=16), ImageFont.load_default(size=15)
    d.text((left, 24), "Same inputs, different numbers of denoising steps (PSNR to the true next frame under each prediction)", font=f_s, fill=muted, anchor="ls")
    for j, c in enumerate(cols):
        d.text((left + j * (tile + gap) + tile // 2, top - 12), c, font=f_h, fill=amber if "demo" in c else text, anchor="ms")
    for i in range(a.n):
        y = top + i * (tile + gap + 24)
        row = [ctx_last[i], target[i], preds[1][i], preds[2][i], preds[3][i]]
        for j, fr in enumerate(row):
            x = left + j * (tile + gap)
            img.paste(to_pil(fr, tile), (x, y))
            if j >= 2:
                d.text((x + tile // 2, y + tile + 17), f"{psnr[j - 1][i]:.1f} dB", font=f_s, fill=amber if j == 3 else muted, anchor="ms")
    pathlib.Path(a.out).parent.mkdir(exist_ok=True)
    img.save(a.out)
    out = {"script": "python tools/sampler_figure.py " + " ".join(sys.argv[1:]), "checkpoint_step": step, "window_indices": idx.tolist(),
           "ctx_sigma": a.ctx_sigma, "init": a.init,
           "psnr_db": {f"euler{k}": [round(float(x), 2) for x in v] for k, v in psnr.items()},
           "mean_psnr_db": {f"euler{k}": round(float(np.mean(v)), 2) for k, v in psnr.items()}}
    pathlib.Path(a.json).write_text(json.dumps(out, indent=1))
    print(json.dumps(out["mean_psnr_db"]), "->", a.out)


if __name__ == "__main__":
    main()
