"""EDM-preconditioned, action-conditioned denoiser (DIAMOND-style) with GameNGen-style context noise.

    D(x; sigma) = c_skip x + c_out F(c_in x; c_noise | context frames, actions, context-noise level)

* Preconditioning and loss weighting: Karras et al. 2022 (EDM), exactly as in DIAMOND's denoiser.py
  (sigma_data = 0.5, log-normal training sigma, offset noise 0.3 folded into the conditioners).
* The conditioning is the last K = 4 frames (channel concat) and the last K actions (embedded, concatenated),
  added to the noise-level embedding and fed to every residual block through adaptive GroupNorm.
* Context-noise augmentation (GameNGen, Valevski et al. 2024): during training the context frames are noised
  with a random sigma_c and the network is told sigma_c; at inference a small fixed sigma_c is used, which lets
  the model correct its own accumulated errors instead of treating them as ground truth.
"""
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from model.unet import Conv3x3, FourierFeatures, GroupNorm, UNet

NUM_ACTION_DIMS = 9          # one-hot(move: 3) + one-hot(strafe: 3) + one-hot(turn: 3)
CTX_NOISE_INPUT_SCALE = 4.0  # sigma_c (<= ~0.5) is multiplied by this before the embedding MLP


def encode_actions(actions: Tensor) -> Tensor:
    """int [..., 3] (move, strafe, turn; each 0..2) -> float one-hot [..., 9]."""
    a = actions.long()
    return torch.cat([F.one_hot(a[..., i], 3) for i in range(3)], dim=-1).float()


@dataclass
class DenoiserConfig:
    channels: Tuple[int, ...] = (64, 64, 64, 64)
    depths: Tuple[int, ...] = (2, 2, 2, 2)
    attn: Tuple[bool, ...] = (False, False, False, False)
    cond_channels: int = 256
    num_ctx: int = 4
    img_channels: int = 3
    sigma_data: float = 0.5
    sigma_offset_noise: float = 0.3
    p_mean: float = -0.4
    p_std: float = 1.2
    sigma_min: float = 2e-3
    sigma_max_train: float = 20.0
    ctx_noise_max: float = 0.5
    ctx_noise_prob: float = 0.9


@dataclass
class Conditioners:
    c_in: Tensor
    c_out: Tensor
    c_skip: Tensor
    c_noise: Tensor


class InnerModel(nn.Module):
    def __init__(self, cfg: DenoiserConfig) -> None:
        super().__init__()
        K, C = cfg.num_ctx, cfg.cond_channels
        assert C % K == 0
        self.cfg = cfg
        self.noise_emb = FourierFeatures(C)
        self.act_emb = nn.Linear(NUM_ACTION_DIMS, C // K, bias=False)      # == sum of three embeddings (one-hot input)
        self.ctx_emb = nn.Sequential(nn.Linear(1, C), nn.SiLU(), nn.Linear(C, C))
        self.cond_proj = nn.Sequential(nn.Linear(C, C), nn.SiLU(), nn.Linear(C, C))
        self.conv_in = Conv3x3((K + 1) * cfg.img_channels, cfg.channels[0])
        self.unet = UNet(C, list(cfg.depths), list(cfg.channels), list(cfg.attn))
        self.norm_out = GroupNorm(cfg.channels[0])
        self.conv_out = Conv3x3(cfg.channels[0], cfg.img_channels)
        nn.init.zeros_(self.conv_out.weight)

    def forward(self, x_noisy: Tensor, c_noise: Tensor, ctx: Tensor, act: Tensor, s_c: Tensor) -> Tensor:
        """x_noisy [B,3,H,W] (scaled by c_in), c_noise [B], ctx [B,K*3,H,W] (scaled by 1/sigma_data),
        act [B,K,9] one-hot, s_c [B] context noise level (sigma units)."""
        cond = self.noise_emb(c_noise) + self.act_emb(act).flatten(1) + self.ctx_emb(s_c.unsqueeze(1) * CTX_NOISE_INPUT_SCALE)
        cond = self.cond_proj(cond)
        x = self.conv_in(torch.cat((ctx, x_noisy), dim=1))
        x = self.unet(x, cond)
        return self.conv_out(F.silu(self.norm_out(x)))


class Denoiser(nn.Module):
    def __init__(self, cfg: DenoiserConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.inner = InnerModel(cfg)

    # -------------------------------------------------------------- preconditioning
    def conditioners(self, sigma: Tensor) -> Conditioners:
        sd = self.cfg.sigma_data
        sigma = (sigma ** 2 + self.cfg.sigma_offset_noise ** 2).sqrt()
        c_in = 1 / (sigma ** 2 + sd ** 2).sqrt()
        c_skip = sd ** 2 / (sigma ** 2 + sd ** 2)
        c_out = sigma * c_skip.sqrt()
        c_noise = sigma.log() / 4
        return Conditioners(c_in, c_out, c_skip, c_noise)

    def _F(self, x, cs: Conditioners, ctx, act, s_c):
        B, K, C, H, W = ctx.shape
        v = lambda t: t.view(-1, 1, 1, 1)
        return self.inner(x * v(cs.c_in), cs.c_noise, ctx.reshape(B, K * C, H, W) / self.cfg.sigma_data, act, s_c)

    def denoise(self, x: Tensor, sigma: Tensor, ctx: Tensor, act: Tensor, s_c: Tensor, clamp: bool = True) -> Tensor:
        """D(x; sigma | ctx, act). x [B,3,H,W], sigma [B], ctx [B,K,3,H,W] in [-1,1] (possibly noised), act [B,K,9]."""
        cs = self.conditioners(sigma)
        d = cs.c_skip.view(-1, 1, 1, 1) * x + cs.c_out.view(-1, 1, 1, 1) * self._F(x, cs, ctx, act, s_c)
        return d.clamp(-1, 1) if clamp else d

    def denoise_flat(self, x: Tensor, sigma: Tensor, ctx_flat: Tensor, act: Tensor, s_c: Tensor) -> Tensor:
        """Same as denoise (clamped) but with the K context frames already stacked on the channel axis
        [B, K*3, H, W]: the interface of the exported ONNX graph."""
        B, KC, H, W = ctx_flat.shape
        return self.denoise(x, sigma, ctx_flat.view(B, KC // x.shape[1], x.shape[1], H, W), act, s_c)

    # -------------------------------------------------------------- training
    def augment_context(self, ctx: Tensor) -> Tuple[Tensor, Tensor]:
        """Noise the context frames with a per-sample level sigma_c ~ U(0, ctx_noise_max) (prob ctx_noise_prob)."""
        B = ctx.shape[0]
        c = self.cfg
        s_c = torch.rand(B, device=ctx.device) * c.ctx_noise_max
        s_c = s_c * (torch.rand(B, device=ctx.device) < c.ctx_noise_prob).float()
        if c.ctx_noise_max <= 0:
            return ctx, torch.zeros(B, device=ctx.device)
        return ctx + torch.randn_like(ctx) * s_c.view(B, 1, 1, 1, 1), s_c

    def loss(self, ctx: Tensor, target: Tensor, act: Tensor):
        """ctx [B,K,3,H,W], target [B,3,H,W], act [B,K,9]; all in [-1,1]."""
        B, C = target.shape[:2]
        c = self.cfg
        sigma = (torch.randn(B, device=target.device) * c.p_std + c.p_mean).exp().clip(c.sigma_min, c.sigma_max_train)
        offset = c.sigma_offset_noise * torch.randn(B, C, 1, 1, device=target.device)
        noisy = target + offset + torch.randn_like(target) * sigma.view(B, 1, 1, 1)
        ctx_n, s_c = self.augment_context(ctx)
        cs = self.conditioners(sigma)
        out = self._F(noisy, cs, ctx_n, act, s_c)
        tgt = (target - cs.c_skip.view(-1, 1, 1, 1) * noisy) / cs.c_out.view(-1, 1, 1, 1)
        loss = F.mse_loss(out, tgt)
        return loss, {"loss": loss.detach()}
