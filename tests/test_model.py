"""Tests for the EDM denoiser, U-Net, sampler and rollout (model/)."""
import math

import numpy as np
import pytest
import torch

from model.denoiser import Denoiser, DenoiserConfig, encode_actions
from model.sampler import build_sigmas, sample_next, rollout
from model import metrics

TINY = dict(channels=(16, 16, 16, 16), depths=(1, 1, 1, 1), cond_channels=64)


def tiny(**kw):
    return Denoiser(DenoiserConfig(**{**TINY, **kw}))


def rand_batch(B=4, K=4, H=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    ctx = torch.rand(B, K, 3, H, H, generator=g) * 2 - 1
    tgt = torch.rand(B, 3, H, H, generator=g) * 2 - 1
    act = encode_actions(torch.randint(0, 3, (B, K, 3), generator=g))
    return ctx, tgt, act


# ------------------------------------------------------------------ schedule / conditioners

def test_sigma_schedule_matches_edm_karras():
    s = build_sigmas(3, 2e-3, 5.0, 7.0)
    assert s.shape == (4,)
    assert s[0] == pytest.approx(5.0, rel=1e-5) and s[-1] == 0.0
    inv = lambda v: v ** (1 / 7)
    expect_mid = (inv(5.0) + 0.5 * (inv(2e-3) - inv(5.0))) ** 7           # Karras et al. 2022, eq. 5, rho = 7
    assert s[1] == pytest.approx(expect_mid, rel=1e-4) and s[2] == pytest.approx(2e-3, rel=1e-4)
    assert (s[:-1] > s[1:]).all()


def test_conditioners_follow_edm_without_offset_noise():
    m = tiny(sigma_offset_noise=0.0)
    sig = torch.tensor([0.1, 1.0, 5.0])
    c = m.conditioners(sig)
    sd = 0.5
    assert torch.allclose(c.c_in, 1 / (sig ** 2 + sd ** 2).sqrt())
    assert torch.allclose(c.c_skip, sd ** 2 / (sig ** 2 + sd ** 2))
    assert torch.allclose(c.c_out, sig * sd / (sig ** 2 + sd ** 2).sqrt())
    assert torch.allclose(c.c_noise, sig.log() / 4)


# ------------------------------------------------------------------ network

def test_default_model_is_4_to_10_million_parameters():
    n = sum(p.numel() for p in Denoiser(DenoiserConfig()).parameters())
    assert 4e6 <= n <= 10e6, n


def test_zero_initialised_output_means_denoiser_starts_as_the_skip_connection_plus_bias():
    m = tiny()
    ctx, tgt, act = rand_batch()
    x = torch.randn(4, 3, 64, 64)
    sig = torch.full((4,), 1.0)
    out = m.denoise(x, sig, ctx, act, torch.zeros(4), clamp=False)
    c = m.conditioners(sig)
    bias = m.inner.conv_out.bias.detach().view(1, 3, 1, 1)            # the output conv weight is zero, its bias is not
    expect = c.c_skip.view(-1, 1, 1, 1) * x + c.c_out.view(-1, 1, 1, 1) * bias
    assert torch.allclose(out, expect, atol=1e-6)


def _randomise_zero_inits(m):
    """DIAMOND zero-initialises the last conv of each residual block, the attention output and the output
    conv, so a fresh network ignores its conditioning; randomise them to test the wiring."""
    from model.unet import ResBlock, SelfAttention2d
    torch.nn.init.normal_(m.inner.conv_out.weight, std=0.05)
    for mod in m.modules():
        if isinstance(mod, ResBlock):
            torch.nn.init.normal_(mod.conv2.weight, std=0.05)
        if isinstance(mod, SelfAttention2d):
            torch.nn.init.normal_(mod.out_proj.weight, std=0.05)


def test_network_is_conditioned_on_actions_noise_level_and_context_noise():
    torch.manual_seed(0)
    m = tiny()
    _randomise_zero_inits(m)
    ctx, tgt, act = rand_batch()
    x = torch.randn(4, 3, 64, 64)
    sig = torch.full((4,), 1.0)
    base = m.denoise(x, sig, ctx, act, torch.zeros(4), clamp=False)
    act2 = act.clone().roll(1, dims=2)
    assert (m.denoise(x, sig, ctx, act2, torch.zeros(4), clamp=False) - base).abs().max() > 1e-4
    assert (m.denoise(x, sig * 3, ctx, act, torch.zeros(4), clamp=False) - base).abs().max() > 1e-4
    assert (m.denoise(x, sig, ctx, act, torch.full((4,), 0.3), clamp=False) - base).abs().max() > 1e-4
    assert (m.denoise(x, sig, ctx * 0.5, act, torch.zeros(4), clamp=False) - base).abs().max() > 1e-4


def real_batch(B=8, seed=0):
    from game import raycaster as rc
    from data.loader import TrajectorySource
    b = TrajectorySource(rc.World(), B, 5, seed=seed).next_batch()
    f = torch.from_numpy(b["frames"]).permute(0, 1, 4, 2, 3).float() / 127.5 - 1
    return f[:, :4], f[:, 4], encode_actions(torch.from_numpy(b["actions"]))


def test_loss_is_finite_and_overfits_a_single_batch_of_game_frames():
    torch.manual_seed(0)
    m = tiny(ctx_noise_max=0.0)
    ctx, tgt, act = real_batch(B=8)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
    first = None
    for i in range(80):
        loss, logs = m.loss(ctx, tgt, act)
        assert torch.isfinite(loss)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i == 0:
            first = loss.item()
    assert loss.item() < 0.75 * first, (first, loss.item())


def test_context_noise_augmentation_changes_the_context_seen_by_the_net():
    torch.manual_seed(1)
    m = tiny(ctx_noise_max=0.5, ctx_noise_prob=1.0)
    ctx, tgt, act = rand_batch()
    noisy, s_c = m.augment_context(ctx)
    assert s_c.shape == (4,) and (s_c >= 0).all() and (s_c <= 0.5).all()
    assert (noisy - ctx).abs().mean() > 0.01
    m0 = tiny(ctx_noise_max=0.5, ctx_noise_prob=0.0)
    noisy0, s0 = m0.augment_context(ctx)
    assert torch.equal(noisy0, ctx) and (s0 == 0).all()


# ------------------------------------------------------------------ sampler

def oracle(x0):
    def f(x, sigma, ctx, act, s_c):
        return x0
    return f


@pytest.mark.parametrize("steps,order", [(1, 1), (2, 1), (3, 1), (2, 2), (3, 2)])
def test_sampler_recovers_x0_for_a_perfect_denoiser(steps, order):
    torch.manual_seed(0)
    x0 = torch.rand(2, 3, 64, 64) * 2 - 1
    sig = build_sigmas(steps, 2e-3, 5.0, 7.0)
    ctx = torch.zeros(2, 4, 3, 64, 64)
    out = sample_next(oracle(x0), ctx, None, torch.zeros(2), sig, order=order, quantize=False)
    assert torch.allclose(out, x0, atol=1e-5)


def test_heun_uses_one_extra_model_call_per_non_final_step():
    calls = []

    def f(x, sigma, ctx, act, s_c):
        calls.append(float(sigma[0]))
        return torch.zeros_like(x)

    sig = build_sigmas(3, 2e-3, 5.0, 7.0)
    sample_next(f, torch.zeros(1, 4, 3, 64, 64), None, torch.zeros(1), sig, order=1, quantize=False)
    assert len(calls) == 3
    calls.clear()
    sample_next(f, torch.zeros(1, 4, 3, 64, 64), None, torch.zeros(1), sig, order=2, quantize=False)
    assert len(calls) == 5          # 3 Euler-equivalent + 2 corrector evaluations (none on the last step to sigma=0)


def test_quantised_sampler_output_is_on_the_8_bit_grid():
    x0 = torch.rand(1, 3, 64, 64) * 2 - 1
    sig = build_sigmas(2, 2e-3, 5.0, 7.0)
    out = sample_next(oracle(x0), torch.zeros(1, 4, 3, 64, 64), None, torch.zeros(1), sig, quantize=True)
    b = (out + 1) / 2 * 255
    assert torch.allclose(b, b.round(), atol=1e-4)


# ------------------------------------------------------------------ rollout

def test_rollout_aligns_actions_and_context_like_training():
    """Generated frame t is conditioned on frames [t .. t+K-1] and actions [t .. t+K-1]."""
    B, K, T = 2, 4, 6
    seen = []

    def fake(x, sigma, ctx, act, s_c):
        seen.append((ctx.clone(), act.clone()))
        # "predict" the mean of the last context frame, so each generated frame is traceable
        return ctx[:, -1] * 0 + ctx[:, -1].mean((1, 2, 3), keepdim=True) * 0 + (len(seen)) / 10.0

    frames = torch.arange(K, dtype=torch.float32).view(1, K, 1, 1, 1).expand(B, K, 3, 64, 64) / 10.0
    acts = torch.randint(0, 3, (B, K + T - 1, 3))
    out = rollout(fake, frames, acts, steps=1, order=1, ctx_sigma=0.0, quantize=False, clamp_unit=False)
    assert out.shape == (B, T, 3, 64, 64)
    ohe = encode_actions(acts)
    for t, (ctx, act) in enumerate(seen):
        assert torch.equal(act, ohe[:, t:t + K])
        if t >= 1:     # the context at step t ends with the frame generated at step t-1
            assert torch.allclose(ctx[:, -1], out[:, t - 1], atol=1e-6)


# ------------------------------------------------------------------ metrics

def test_psnr_and_ssim_basics():
    a = torch.rand(3, 3, 64, 64)
    assert metrics.psnr(a, a).isinf().all()
    b = (a + 0.1 * torch.randn_like(a)).clamp(0, 1)
    p = metrics.psnr(a, b)
    assert (p > 10).all() and (p < 40).all()
    s_same = metrics.ssim(a, a)
    s_noisy = metrics.ssim(a, b)
    assert torch.allclose(s_same, torch.ones(3), atol=1e-4)
    assert (s_noisy < s_same).all()


def test_group_count_divides_any_channel_width():
    from model.unet import num_groups
    assert [num_groups(c) for c in (32, 64, 96, 128, 192)] == [1, 2, 3, 4, 6]       # DIAMOND: channels // 32
    for c in (24, 40, 48, 56, 80, 112, 144):
        assert c % num_groups(c) == 0
    m = Denoiser(DenoiserConfig(channels=(48, 64, 96, 128), depths=(1, 1, 1, 1), cond_channels=64))
    ctx, tgt, act = rand_batch(B=2)
    loss, _ = m.loss(ctx, tgt, act)
    assert torch.isfinite(loss)
