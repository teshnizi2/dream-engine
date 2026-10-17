"""Evaluation helpers: fixed validation windows, 1-step prediction PSNR/SSIM and autoregressive-rollout drift
curves against the true game. `denoise_fn(x, sigma, ctx, act, s_c)` is `Denoiser.denoise` (or any stand-in)."""
from typing import Callable, Dict, Optional

import numpy as np
import torch

from data.loader import TrajectorySource
from model import metrics
from model.denoiser import encode_actions
from model.sampler import build_sigmas, randn_with, rollout, sample_next


def _to_unit(frames_u8: np.ndarray) -> torch.Tensor:
    """uint8 [..., H, W, 3] -> float32 [..., 3, H, W] in [-1, 1]."""
    t = torch.from_numpy(np.ascontiguousarray(frames_u8)).float() / 127.5 - 1
    return t.movedim(-1, -3).contiguous()


def val_windows(world, n: int = 256, seed: int = 999, K: int = 4) -> Dict[str, torch.Tensor]:
    """n fixed independent (K + 1)-frame windows (K context frames + target) from the exploration policy. The same seed
    gives the same games for every K (a longer window just extends the shorter one)."""
    b = TrajectorySource(world, n, K + 1, seed=seed).next_batch()
    return {"frames": _to_unit(b["frames"]), "actions": torch.from_numpy(b["actions"])}


def eval_trajectories_t(world, n: int, length: int, seed: int = 123) -> Dict[str, torch.Tensor]:
    b = TrajectorySource(world, n, length, seed=seed, warmup_max=0).next_batch()
    return {"frames": _to_unit(b["frames"]), "actions": torch.from_numpy(b["actions"]), "poses": torch.from_numpy(b["poses"])}


def wake_slice(traj: Dict[str, torch.Tensor], wake: int, K: int) -> Dict[str, torch.Tensor]:
    """The part of a trajectory a K-context model needs when it wakes up at real frame `wake`: the K real frames before
    it, then everything after. Models with different K are thus scored on the same real frames (wake, wake+1, ...)."""
    assert wake >= K, "not enough real frames before the wake-up frame"
    out = {"frames": traj["frames"][:, wake - K:], "actions": traj["actions"][:, wake - K:]}
    if "poses" in traj:
        out["poses"] = traj["poses"][:, wake - K:]
    return out


def _gen(seed: int, device) -> torch.Generator:
    g = torch.Generator(device=device if device.type != "mps" else "cpu")
    g.manual_seed(seed)
    return g


@torch.no_grad()
def one_step_metrics(denoise_fn: Callable, windows: Dict[str, torch.Tensor], steps: int = 3, order: int = 1,
                     ctx_sigma: float = 0.1, init: str = "edm", seed: int = 0, batch: int = 64,
                     sigma_max: float = 5.0, return_pred: bool = False, K: Optional[int] = None):
    """Predict the last frame of every window from the K frames before it + the K actions ending at it; PSNR/SSIM vs the
    true next frame (8-bit). K defaults to the window length - 1 (4 for the 5-frame windows of v2); a K smaller than that
    uses the last K context frames, so models with different K are scored on the same target frames."""
    fr, ac = windows["frames"], windows["actions"]
    T = fr.shape[1]
    K = T - 1 if K is None else K
    assert 1 <= K <= T - 1
    dev = fr.device
    sig = build_sigmas(steps, 2e-3, sigma_max, 7.0)
    ps, ss, preds = [], [], []
    gen = None                                              # one generator per call, so a seed reproduces the context noise and the sampler noise
    for i in range(0, len(fr), batch):
        ctx, tgt, a = fr[i:i + batch, T - 1 - K:T - 1], fr[i:i + batch, T - 1], ac[i:i + batch, T - 1 - K:T - 1]
        dev = _device_of(denoise_fn, ctx)
        ctx, tgt = ctx.to(dev), tgt.to(dev)
        gen = gen if gen is not None else _gen(seed, dev)
        c = ctx
        if ctx_sigma > 0:
            c = c + randn_with(c.shape, gen, dev, c.dtype) * ctx_sigma
        s_c = torch.full((len(ctx),), float(ctx_sigma), device=dev)
        pred = sample_next(denoise_fn, c, encode_actions(a).to(dev), s_c, sig.to(dev), order=order,
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


def make_frame_bank(world, n: int = 20000, seed: int = 777):
    """A bank of real game frames from the exploration policy (the reference set for `nearest_real_psnr`)."""
    src = TrajectorySource(world, 256, 2, seed=seed, warmup_max=30)
    frames = []
    while sum(len(f) for f in frames) < n:
        frames.append(src.next_batch()["frames"][:, 1])
    f = np.concatenate(frames)[:n]
    return {"frames": _to_unit(f)}


@torch.no_grad()
def nearest_real_psnr(frames: torch.Tensor, bank, chunk: int = 512) -> torch.Tensor:
    """For each frame [M, 3, 64, 64] in [-1,1]: PSNR (dB, 8-bit range) to its nearest neighbour in the bank of real
    frames. High = the frame looks like a frame of the real game (even if it is not the true next frame)."""
    dev = frames.device
    b = bank.get("flat")
    if b is None or b.device != dev:
        b = ((bank["frames"].to(dev) + 1) / 2).flatten(1)
        bank["flat"] = b
    bn = (b * b).sum(1)
    out = []
    for i in range(0, len(frames), chunk):
        a = ((frames[i:i + chunk] + 1) / 2).flatten(1).to(dev)
        d2 = (a * a).sum(1, keepdim=True) + bn[None] - 2 * a @ b.T
        mse = d2.min(1).values.clamp(min=0) / a.shape[1]
        out.append(10 * torch.log10(1.0 / mse.clamp(min=1e-10)))
    return torch.cat(out).clamp(max=100).cpu()
