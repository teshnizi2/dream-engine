"""PSNR and SSIM on image batches in [0, 1], shape [B, 3, H, W]. SSIM: 11x11 Gaussian window (sigma 1.5),
averaged over channels, as in Wang et al. 2004."""
import torch
import torch.nn.functional as F


def psnr(a: torch.Tensor, b: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    mse = ((a - b) ** 2).flatten(1).mean(1)
    return 10 * torch.log10(max_val ** 2 / mse)


def _gauss(size=11, sigma=1.5, device=None, dtype=torch.float32):
    x = torch.arange(size, device=device, dtype=dtype) - size // 2
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g[:, None] * g[None, :]


def ssim(a: torch.Tensor, b: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    C = a.shape[1]
    w = _gauss(device=a.device, dtype=a.dtype).expand(C, 1, 11, 11).contiguous()
    f = lambda x: F.conv2d(x, w, groups=C)
    mu_a, mu_b = f(a), f(b)
    saa = f(a * a) - mu_a ** 2
    sbb = f(b * b) - mu_b ** 2
    sab = f(a * b) - mu_a * mu_b
    c1, c2 = (0.01 * max_val) ** 2, (0.03 * max_val) ** 2
    s = ((2 * mu_a * mu_b + c1) * (2 * sab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (saa + sbb + c2))
    return s.flatten(1).mean(1)
