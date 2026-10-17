"""EDM sampler (Euler = DIAMOND's default, or Heun) and autoregressive rollouts."""
from typing import Callable, Optional

import torch
from torch import Tensor

from model.denoiser import encode_actions

DenoiseFn = Callable[[Tensor, Tensor, Tensor, Tensor, Tensor], Tensor]   # (x, sigma[B], ctx, act, s_c) -> D


def build_sigmas(num_steps: int, sigma_min: float = 2e-3, sigma_max: float = 5.0, rho: float = 7.0,
                 device: Optional[torch.device] = None) -> Tensor:
    """Karras et al. schedule, length num_steps + 1 (the last entry is 0)."""
    min_inv, max_inv = sigma_min ** (1 / rho), sigma_max ** (1 / rho)
    l = torch.linspace(0, 1, num_steps, device=device)
    sigmas = (max_inv + l * (min_inv - max_inv)) ** rho
    return torch.cat((sigmas, sigmas.new_zeros(1)))


def quantize_unit(x: Tensor) -> Tensor:
    """Round to the 8-bit grid, staying in [-1, 1]."""
    return x.clamp(-1, 1).add(1).div(2).mul(255).round().div(255).mul(2).sub(1)


@torch.no_grad()
def sample_next(denoise: DenoiseFn, ctx: Tensor, act: Optional[Tensor], s_c: Tensor, sigmas: Tensor,
                order: int = 1, quantize: bool = True, generator: Optional[torch.Generator] = None,
                init: str = "edm") -> Tensor:
    """Sample the next frame. ctx [B,K,3,H,W] (already noised if desired), act [B,K,9], s_c [B].
    order 1 = Euler (one model call per step), order 2 = Heun (an extra call on every step except the last).
    init: 'edm' starts from sigma_max * N(0, I) (Karras); 'unit' from N(0, I) as in the DIAMOND code."""
    B, _, C, H, W = ctx.shape
    x = torch.randn(B, C, H, W, generator=generator, device=ctx.device, dtype=ctx.dtype)
    if init == "edm":
        x = x * sigmas[0]
    for i in range(len(sigmas) - 1):
        sigma, nsigma = sigmas[i], sigmas[i + 1]
        den = denoise(x, sigma.expand(B), ctx, act, s_c)
        d = (x - den) / sigma
        dt = nsigma - sigma
        if order == 1 or nsigma == 0:
            x = x + d * dt
        else:
            x2 = x + d * dt
            den2 = denoise(x2, nsigma.expand(B), ctx, act, s_c)
            d2 = (x2 - den2) / nsigma
            x = x + (d + d2) / 2 * dt
    x = x.clamp(-1, 1)
    return quantize_unit(x) if quantize else x


@torch.no_grad()
def rollout(denoise: DenoiseFn, init_frames: Tensor, actions: Tensor, steps: int = 3, order: int = 1,
            ctx_sigma: float = 0.1, sigma_max: float = 5.0, quantize: bool = True,
            generator: Optional[torch.Generator] = None, clamp_unit: bool = True, init: str = "edm") -> Tensor:
    """Pure-imagination rollout. init_frames [B,K,3,H,W] in [-1,1] (real frames f_0..f_{K-1}),
    actions int [B, K+T-1, 3] where actions[:, i] leads frame i -> i+1. Returns the T generated frames
    f_K .. f_{K+T-1}, each conditioned on the previous K frames and the K actions ending at its own."""
    B, K = init_frames.shape[:2]
    T = actions.shape[1] - K + 1
    ohe = encode_actions(actions).to(init_frames.device)
    sigmas = build_sigmas(steps, 2e-3, sigma_max, 7.0, device=init_frames.device)
    ctx = init_frames
    s_c = torch.full((B,), float(ctx_sigma), device=init_frames.device)
    outs = []
    for t in range(T):
        c = ctx
        if ctx_sigma > 0:
            c = c + torch.randn(c.shape, generator=generator, device=c.device, dtype=c.dtype) * ctx_sigma
        nxt = sample_next(denoise, c, ohe[:, t:t + K], s_c, sigmas, order=order,
                          quantize=quantize, generator=generator, init=init)
        if not clamp_unit:
            pass
        outs.append(nxt)
        ctx = torch.cat([ctx[:, 1:], nxt.unsqueeze(1)], dim=1)
    return torch.stack(outs, 1)
