#!/usr/bin/env python
"""Train the action-conditioned EDM denoiser (DIAMOND-style) on the raycaster maze, with infinite
on-the-fly data. Resumable: re-run the same command with --resume and it continues from ckpt_last.pt
(the LR schedule is a function of the global step, so chained 4 h jobs behave like one long run).

    python train.py --out runs/smoke --steps 1500 --batch 32 --device mps --time-limit-min 18
    python train.py --out runs/full --steps 200000 --batch 128 --device cuda --amp bf16 --resume
"""
import argparse
import json
import math
import os
import pathlib
import signal
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from game import raycaster as rc  # noqa: E402
from data.loader import BatchStream  # noqa: E402
from model import evaluate as ev  # noqa: E402
from model.denoiser import Denoiser, DenoiserConfig, encode_actions  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=2000, help="total training steps of the whole (possibly chained) run")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--wd", type=float, default=1e-2)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--min-lr-frac", type=float, default=0.1)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--channels", default="64,64,64,64")
    p.add_argument("--depths", default="2,2,2,2")
    p.add_argument("--cond", type=int, default=256)
    p.add_argument("--ctx-noise-max", type=float, default=0.5)
    p.add_argument("--ctx-noise-prob", type=float, default=0.9)
    p.add_argument("--ctx-blur-prob", type=float, default=0.0)
    p.add_argument("--ctx-blur-max", type=float, default=1.5)
    p.add_argument("--unroll", type=int, default=1, help="DIAMOND-style multi-step training: later steps see the model's own denoised frames")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--prefetch", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", default="none", choices=["none", "bf16", "fp16"])
    p.add_argument("--compile", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--init-from", default=None, help="start from the EMA weights of another run's checkpoint (fresh optimiser, step 0)")
    p.add_argument("--eval-every", type=int, default=500, help="0 disables evaluation")
    p.add_argument("--eval-n", type=int, default=64)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument("--milestone-every", type=int, default=0, help="also keep ckpt_stepN.pt every N steps (0 = never)")
    p.add_argument("--time-limit-min", type=float, default=0.0, help="stop gracefully (with a checkpoint) after this long")
    return p.parse_args(argv)


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def lr_at(step, a):
    if step < a.warmup:
        return a.lr * (step + 1) / a.warmup
    t = min(1.0, (step - a.warmup) / max(1, a.steps - a.warmup))
    return a.lr * (a.min_lr_frac + (1 - a.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * t)))


def save_atomic(obj, path):
    tmp = str(path) + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def to_batch(b, device):
    """-> all frames [B, T, 3, H, W] in [-1, 1] and one-hot actions [B, T-1, 9]."""
    f = b["frames"].to(device, non_blocking=True).permute(0, 1, 4, 2, 3).float().div(127.5).sub(1)
    return f, encode_actions(b["actions"].to(device))


@torch.no_grad()
def sample_grid(model, windows, path, n=8, steps=3, ctx_sigma=0.1):
    """PNG: rows of [last context | truth | prediction | 4x abs error]."""
    from PIL import Image
    dev = next(model.parameters()).device
    w = {k: v[:n] for k, v in windows.items()}
    m = ev.one_step_metrics(model.denoise, w, steps=steps, ctx_sigma=ctx_sigma, return_pred=True)
    rows = []
    for i in range(n):
        ctx = w["frames"][i, 3]
        gt = w["frames"][i, 4]
        pr = m["pred"][i]
        err = ((pr - gt).abs() * 2 - 1).clamp(-1, 1)
        row = torch.cat([ctx, gt, pr, err], dim=2)                   # [3, 64, 256]
        rows.append(row)
    img = torch.cat(rows, dim=1)                                      # [3, 64n, 256]
    img = ((img + 1) / 2 * 255).round().clamp(0, 255).byte().permute(1, 2, 0).numpy()
    Image.fromarray(img).resize((img.shape[1] * 3, img.shape[0] * 3), Image.NEAREST).save(path)


def main(argv=None):
    a = parse_args(argv)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device(a.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    cfg = DenoiserConfig(channels=tuple(int(c) for c in a.channels.split(",")),
                         depths=tuple(int(c) for c in a.depths.split(",")),
                         attn=tuple(False for _ in a.channels.split(",")), cond_channels=a.cond,
                         ctx_noise_max=a.ctx_noise_max, ctx_noise_prob=a.ctx_noise_prob,
                         ctx_blur_prob=a.ctx_blur_prob, ctx_blur_max=a.ctx_blur_max)
    torch.manual_seed(a.seed)
    model = Denoiser(cfg).to(device)
    ema = Denoiser(cfg).to(device)
    if a.init_from and not (a.resume and (out / "ckpt_last.pt").exists()):
        src = torch.load(a.init_from, map_location=device, weights_only=False)
        model.load_state_dict(src["ema"])
        print(f"initialised from EMA weights of {a.init_from} (step {src.get('step')})", flush=True)
    ema.load_state_dict(model.state_dict())
    ema.eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd, eps=1e-8)
    step, resumes, elapsed_prev = 0, 0, 0.0
    ck_path = out / "ckpt_last.pt"
    if a.resume and ck_path.exists():
        ck = torch.load(ck_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        step, resumes, elapsed_prev = ck["step"], ck.get("resumes", 0) + 1, ck.get("elapsed_s", 0.0)
        print(f"resumed from step {step} (resume #{resumes})", flush=True)
        if step >= a.steps:
            print("run already complete (step >= --steps); nothing to do", flush=True)
            return
    n_params = sum(p.numel() for p in model.parameters())
    (out / "config.json").write_text(json.dumps({"denoiser": cfg.__dict__, "args": vars(a), "params": n_params,
                                                 "device": str(device)}, indent=1, default=list))
    print(f"device {device} | params {n_params / 1e6:.2f}M | batch {a.batch} | steps {step} -> {a.steps}", flush=True)

    loss_fn = torch.compile(model.loss) if a.compile else model.loss
    loss_unrolled_fn = torch.compile(model.loss_unrolled) if a.compile else model.loss_unrolled
    stream = BatchStream(a.batch, 4 + a.unroll, base_seed=a.seed + 7919 * resumes)
    kw = dict(batch_size=None, num_workers=a.workers)
    if a.workers > 0:
        kw.update(prefetch_factor=a.prefetch, persistent_workers=True)
    loader = iter(DataLoader(stream, **kw))

    world = rc.World()
    val = ev.val_windows(world, n=a.eval_n, seed=999) if a.eval_every else None
    amp_dtype = {"none": None, "bf16": torch.bfloat16, "fp16": torch.float16}[a.amp]
    scaler = torch.amp.GradScaler("cuda") if (a.amp == "fp16" and device.type == "cuda") else None

    stop = {"flag": False}

    def on_signal(signum, frame):
        stop["flag"] = True
        print(f"signal {signum}: finishing the current step, then checkpointing", flush=True)

    for s in (signal.SIGTERM, signal.SIGINT, getattr(signal, "SIGUSR1", signal.SIGTERM)):
        signal.signal(s, on_signal)

    def checkpoint(final=False):
        save_atomic({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "step": step,
                     "config": cfg.__dict__, "args": vars(a), "resumes": resumes,
                     "elapsed_s": elapsed_prev + (time.time() - t0)}, ck_path)
        if a.milestone_every and step % a.milestone_every == 0:
            save_atomic({"ema": ema.state_dict(), "config": cfg.__dict__, "step": step}, out / f"ema_step{step}.pt")
        if final:
            save_atomic({"ema": ema.state_dict(), "config": cfg.__dict__, "step": step}, out / "final_ema.pt")

    log_f = open(out / "metrics.jsonl", "a")
    t0 = time.time()
    t_log, loss_acc, n_acc = time.time(), torch.zeros((), device=device), 0
    ema_p, mod_p = list(ema.parameters()), list(model.parameters())
    last_ck_step = step
    while step < a.steps and not stop["flag"]:
        frames, act = to_batch(next(loader), device)
        for g in opt.param_groups:
            g["lr"] = lr_at(step, a)
        model.train()
        def compute_loss():
            if a.unroll > 1:
                return loss_unrolled_fn(frames, act, a.unroll)
            return loss_fn(frames[:, :4], frames[:, 4], act[:, :4])

        if amp_dtype is not None:
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                loss, _ = compute_loss()
        else:
            loss, _ = compute_loss()
        opt.zero_grad(set_to_none=True)
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
            opt.step()
        step += 1
        decay = min(a.ema, (1 + step) / (10 + step))
        with torch.no_grad():
            torch._foreach_mul_(ema_p, decay)
            torch._foreach_add_(ema_p, mod_p, alpha=1 - decay)
        loss_acc += loss.detach()
        n_acc += 1
        if step % a.log_every == 0 or step == a.steps:
            l = (loss_acc / n_acc).item()
            now = time.time()
            row = {"step": step, "loss": l, "lr": lr_at(step - 1, a), "steps_per_s": n_acc / (now - t_log),
                   "elapsed_s": elapsed_prev + now - t0}
            log_f.write(json.dumps(row) + "\n")
            log_f.flush()
            print(f"step {step:7d} loss {l:.4f} lr {row['lr']:.2e} {row['steps_per_s']:.2f} it/s "
                  f"({row['elapsed_s'] / 60:.1f} min)", flush=True)
            loss_acc.zero_()
            n_acc, t_log = 0, now
        if a.eval_every and step % a.eval_every == 0:
            m = ev.one_step_metrics(ema.denoise, val, steps=3, ctx_sigma=0.1)
            m0 = ev.one_step_metrics(ema.denoise, val, steps=3, ctx_sigma=0.0)
            row = {"step": step, "eval_psnr_ctx0.1": m["psnr"], "eval_ssim_ctx0.1": m["ssim"],
                   "eval_psnr_ctx0": m0["psnr"], "eval_ssim_ctx0": m0["ssim"]}
            log_f.write(json.dumps(row) + "\n")
            log_f.flush()
            print("  eval (EMA, 3 Euler steps): " + " ".join(f"{k} {v:.3f}" for k, v in row.items() if k != "step"), flush=True)
            (out / "samples").mkdir(exist_ok=True)
            sample_grid(ema, val, out / "samples" / f"step_{step:07d}.png")
        if a.ckpt_every and step % a.ckpt_every == 0:
            checkpoint()
            last_ck_step = step
        if a.time_limit_min and (time.time() - t0) / 60 >= a.time_limit_min:
            print(f"time limit {a.time_limit_min} min reached at step {step}", flush=True)
            break
    checkpoint(final=(step >= a.steps))
    print(f"done at step {step}; checkpoint {ck_path}", flush=True)


if __name__ == "__main__":
    main()
