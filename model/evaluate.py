"""Evaluation helpers: fixed validation windows, 1-step prediction PSNR/SSIM and autoregressive-rollout drift
curves against the true game. `denoise_fn(x, sigma, ctx, act, s_c)` is `Denoiser.denoise` (or any stand-in)."""
from typing import Callable, Dict, Optional

import numpy as np
import torch

from data.loader import TrajectorySource
from model import metrics
from model.denoiser import encode_actions
from model.sampler import build_sigmas, rollout, sample_next


def _to_unit(frames_u8: np.ndarray) -> torch.Tensor:
    """uint8 [..., H, W, 3] -> float32 [..., 3, H, W] in [-1, 1]."""
    t = torch.from_numpy(np.ascontiguousarray(frames_u8)).float() / 127.5 - 1
    return t.movedim(-1, -3).contiguous()


def val_windows(world, n: int = 256, seed: int = 999) -> Dict[str, torch.Tensor]:
    """n fixed independent 5-frame windows (4 context frames + target) from the exploration policy."""
    b = TrajectorySource(world, n, 5, seed=seed).next_batch()
    return {"frames": _to_unit(b["frames"]), "actions": torch.from_numpy(b["actions"])}


def eval_trajectories_t(world, n: int, length: int, seed: int = 123) -> Dict[str, torch.Tensor]:
    b = TrajectorySource(world, n, length, seed=seed, warmup_max=0).next_batch()
    return {"frames": _to_unit(b["frames"]), "actions": torch.from_numpy(b["actions"]), "poses": torch.from_numpy(b["poses"])}


def _gen(seed: int, device) -> torch.Generator:
    g = torch.Generator(device=device if device.type != "mps" else "cpu")
    g.manual_seed(seed)
    return g


@torch.no_grad()
def one_step_metrics(denoise_fn: Callable, windows: Dict[str, torch.Tensor], steps: int = 3, order: int = 1,
                     ctx_sigma: float = 0.1, init: str = "edm", seed: int = 0, batch: int = 64,
                     sigma_max: float = 5.0, return_pred: bool = False):
    """Predict frame 5 from frames 1-4 + actions for every window; PSNR/SSIM vs the true next frame (8-bit)."""
    fr, ac = windows["frames"], windows["actions"]
    dev = fr.device
    sig = build_sigmas(steps, 2e-3, sigma_max, 7.0)
    ps, ss, preds = [], [], []
    for i in range(0, len(fr), batch):
        ctx, tgt, a = fr[i:i + batch], fr[i:i + batch, 4], ac[i:i + batch]
        dev = _device_of(denoise_fn, ctx)
        ctx, tgt = ctx[:, :4].to(dev), tgt.to(dev)
        gen = None
        c = ctx
        if ctx_sigma > 0:
            c = c + torch.randn(c.shape, device=dev) * ctx_sigma
        s_c = torch.full((len(ctx),), float(ctx_sigma), device=dev)
        pred = sample_next(denoise_fn, c, encode_actions(a[:, :4]).to(dev), s_c, sig.to(dev), order=order,
                           quantize=True, generator=gen, init=init)
        p01, t01 = (pred + 1) / 2, (tgt + 1) / 2
        ps.append(metrics.psnr(p01, t01).clamp(max=100).cpu())
        ss.append(metrics.ssim(p01, t01).cpu())
        if return_pred:
            preds.append(pred.cpu())
    out = {"psnr": float(torch.cat(ps).mean()), "ssim": float(torch.cat(ss).mean()),
           "psnr_per_window": torch.cat(ps).numpy()}
    if return_pred:
        out["pred"] = torch.cat(preds)
    return out


def _device_of(fn, fallback: torch.Tensor):
    mod = getattr(fn, "__self__", None)
    if mod is not None and hasattr(mod, "parameters"):
        try:
            return next(mod.parameters()).device
        except StopIteration:
            pass
    return fallback.device


@torch.no_grad()
def rollout_metrics(denoise_fn: Callable, traj: Dict[str, torch.Tensor], horizon: int, steps: int = 3, order: int = 1,
                    ctx_sigma: float = 0.1, init: str = "edm", seed: int = 0, batch: int = 16,
                    return_frames: bool = False, K: int = 4):
    """Roll out `horizon` frames in pure imagination from 4 real frames and the real action stream; compare each
    generated frame with the true game frame. Returns per-step mean PSNR / SSIM over episodes."""
    fr, ac = traj["frames"], traj["actions"]
    assert fr.shape[1] >= K + horizon
    ps, ss, gens = [], [], []
    for i in range(0, len(fr), batch):
        dev = _device_of(denoise_fn, fr)
        f = fr[i:i + batch].to(dev)
        a = ac[i:i + batch, :K + horizon - 1].to(dev)
        gen = _gen(seed + i, dev)
        out = rollout(denoise_fn, f[:, :K], a, steps=steps, order=order, ctx_sigma=ctx_sigma, quantize=True,
                      generator=gen, init=init)
        gt = f[:, K:K + horizon]
        B = out.shape[0]
        p = metrics.psnr(((out + 1) / 2).flatten(0, 1), ((gt + 1) / 2).flatten(0, 1)).view(B, horizon).clamp(max=100)
        s = metrics.ssim(((out + 1) / 2).flatten(0, 1), ((gt + 1) / 2).flatten(0, 1)).view(B, horizon)
        ps.append(p.cpu())
        ss.append(s.cpu())
        if return_frames:
            gens.append(out.cpu())
    res = {"psnr": torch.cat(ps).mean(0).numpy(), "ssim": torch.cat(ss).mean(0).numpy(),
           "psnr_all": torch.cat(ps).numpy(), "ssim_all": torch.cat(ss).numpy()}
    if return_frames:
        res["frames"] = torch.cat(gens)
    return res
