"""U-Net building blocks, following DIAMOND (Alonso et al., NeurIPS 2024; github.com/eloialonso/diamond,
src/models/blocks.py, MIT licence): residual blocks conditioned through adaptive GroupNorm, a U-Net with
`n+1` residual blocks per level on the way up (fed by the skip connections), self-attention (head dim 8)
in the middle block, strided-conv downsampling and nearest-upsample + conv upsampling.

Re-written here (not imported) so that the graph exports cleanly to ONNX / onnxruntime-web.
"""
import math
from functools import partial
from typing import List, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

GN_GROUP_SIZE = 32
GN_EPS = 1e-5
ATTN_HEAD_DIM = 8

Conv1x1 = partial(nn.Conv2d, kernel_size=1, stride=1, padding=0)
Conv3x3 = partial(nn.Conv2d, kernel_size=3, stride=1, padding=1)


class GroupNorm(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(max(1, in_channels // GN_GROUP_SIZE), in_channels, eps=GN_EPS)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x)


class AdaGroupNorm(nn.Module):
    """GroupNorm without affine parameters, followed by a per-channel scale and shift predicted from `cond`."""

    def __init__(self, in_channels: int, cond_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_groups = max(1, in_channels // GN_GROUP_SIZE)
        self.linear = nn.Linear(cond_channels, in_channels * 2)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        x = F.group_norm(x, self.num_groups, eps=GN_EPS)
        scale, shift = self.linear(cond)[:, :, None, None].chunk(2, dim=1)
        return x * (1 + scale) + shift


class SelfAttention2d(nn.Module):
    def __init__(self, in_channels: int, head_dim: int = ATTN_HEAD_DIM) -> None:
        super().__init__()
        self.n_head = max(1, in_channels // head_dim)
        assert in_channels % self.n_head == 0
        self.norm = GroupNorm(in_channels)
        self.qkv_proj = Conv1x1(in_channels, in_channels * 3)
        self.out_proj = Conv1x1(in_channels, in_channels)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: Tensor) -> Tensor:
        n, c, h, w = x.shape
        x = self.norm(x)
        qkv = self.qkv_proj(x)
        qkv = qkv.view(n, self.n_head * 3, c // self.n_head, h * w).transpose(2, 3).contiguous()
        q, k, v = qkv.chunk(3, dim=1)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(2, 3).reshape(n, c, h, w)
        return x + self.out_proj(y)


class FourierFeatures(nn.Module):
    """Embedding of a scalar (the noise level) with fixed random frequencies."""

    def __init__(self, cond_channels: int) -> None:
        super().__init__()
        assert cond_channels % 2 == 0
        self.register_buffer("weight", torch.randn(1, cond_channels // 2))

    def forward(self, input: Tensor) -> Tensor:
        f = 2 * math.pi * input.unsqueeze(1) @ self.weight
        return torch.cat([f.cos(), f.sin()], dim=-1)


class Downsample(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)
        nn.init.orthogonal_(self.conv.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = Conv3x3(in_channels, in_channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_channels: int, attn: bool) -> None:
        super().__init__()
        self.proj = Conv1x1(in_channels, out_channels) if in_channels != out_channels else nn.Identity()
        self.norm1 = AdaGroupNorm(in_channels, cond_channels)
        self.conv1 = Conv3x3(in_channels, out_channels)
        self.norm2 = AdaGroupNorm(out_channels, cond_channels)
        self.conv2 = Conv3x3(out_channels, out_channels)
        self.attn = SelfAttention2d(out_channels) if attn else nn.Identity()
        nn.init.zeros_(self.conv2.weight)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        r = self.proj(x)
        x = self.conv1(F.silu(self.norm1(x, cond)))
        x = self.conv2(F.silu(self.norm2(x, cond)))
        return self.attn(x + r)


class ResBlocks(nn.Module):
    def __init__(self, list_in: List[int], list_out: List[int], cond_channels: int, attn: bool) -> None:
        super().__init__()
        assert len(list_in) == len(list_out)
        self.resblocks = nn.ModuleList([ResBlock(i, o, cond_channels, attn) for i, o in zip(list_in, list_out)])

    def forward(self, x: Tensor, cond: Tensor, to_cat: Optional[List[Tensor]] = None):
        outputs = []
        for i, block in enumerate(self.resblocks):
            x = x if to_cat is None else torch.cat((x, to_cat[i]), dim=1)
            x = block(x, cond)
            outputs.append(x)
        return x, outputs


class UNet(nn.Module):
    def __init__(self, cond_channels: int, depths: List[int], channels: List[int], attn_depths: List[bool]) -> None:
        super().__init__()
        assert len(depths) == len(channels) == len(attn_depths)
        self._num_down = len(channels) - 1
        d_blocks, u_blocks = [], []
        for i, n in enumerate(depths):
            c1, c2 = channels[max(0, i - 1)], channels[i]
            d_blocks.append(ResBlocks([c1] + [c2] * (n - 1), [c2] * n, cond_channels, attn_depths[i]))
            u_blocks.append(ResBlocks([2 * c2] * n + [c1 + c2], [c2] * n + [c1], cond_channels, attn_depths[i]))
        self.d_blocks = nn.ModuleList(d_blocks)
        self.u_blocks = nn.ModuleList(reversed(u_blocks))
        self.mid_blocks = ResBlocks([channels[-1]] * 2, [channels[-1]] * 2, cond_channels, attn=True)
        self.downsamples = nn.ModuleList([nn.Identity()] + [Downsample(c) for c in channels[:-1]])
        self.upsamples = nn.ModuleList([nn.Identity()] + [Upsample(c) for c in reversed(channels[:-1])])

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        *_, h, w = x.size()
        n = self._num_down
        ph = math.ceil(h / 2 ** n) * 2 ** n - h
        pw = math.ceil(w / 2 ** n) * 2 ** n - w
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph))
        d_outputs = []
        for block, down in zip(self.d_blocks, self.downsamples):
            x_down = down(x)
            x, block_outputs = block(x_down, cond)
            d_outputs.append((x_down, *block_outputs))
        x, _ = self.mid_blocks(x, cond)
        for block, up, skip in zip(self.u_blocks, self.upsamples, reversed(d_outputs)):
            x_up = up(x)
            x, _ = block(x_up, cond, skip[::-1])
        if ph or pw:
            x = x[..., :h, :w]
        return x
