"""Differentiable image augmentations: the DSA family (Zhao & Bilen, 2021) and a light translate/flip.

``dsa`` draws ONE augmentation family per call from the strategy list; parameters are drawn per sample
(``shared=False``, used when training evaluation networks) or once for the whole batch (``shared=True``, the
"siamese" use in distribution matching). The affine warp is a bilinear sampler built from indexing and weights
only, so it is differentiable with respect to the images on every backend (``grid_sample``'s backward is not
available on Apple MPS)."""
from __future__ import annotations

import math
import random

import torch
import torch.nn.functional as F


def affine_warp(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Bilinear affine warp with zero padding; equals ``F.grid_sample(x, F.affine_grid(theta, x.shape,
    align_corners=True), padding_mode="zeros", align_corners=True)``."""
    n, c, h, w = x.shape
    ys = torch.linspace(-1, 1, h, device=x.device)
    xs = torch.linspace(-1, 1, w, device=x.device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    base = torch.stack([gx, gy, torch.ones_like(gx)], -1).reshape(1, h * w, 3)
    src = base @ theta.transpose(1, 2)
    px = (src[..., 0] + 1) / 2 * (w - 1)
    py = (src[..., 1] + 1) / 2 * (h - 1)
    x0, y0 = px.floor(), py.floor()
    wx1, wy1 = px - x0, py - y0
    wx0, wy0 = 1 - wx1, 1 - wy1
    xf = x.permute(0, 2, 3, 1)
    bi = torch.arange(n, device=x.device)[:, None].expand(n, h * w)
    out = 0
    for dx, dy, wt in [(0, 0, wx0 * wy0), (1, 0, wx1 * wy0), (0, 1, wx0 * wy1), (1, 1, wx1 * wy1)]:
        ix, iy = (x0 + dx).long(), (y0 + dy).long()
        ok = ((ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)).float()
        v = xf[bi, iy.clamp(0, h - 1), ix.clamp(0, w - 1)]
        out = out + v * (wt * ok)[..., None]
    return out.reshape(n, h, w, c).permute(0, 3, 1, 2)


def _shape(n: int, shared: bool):
    return (1,) if shared else (n,)


def aug_scale(x, shared):
    n = x.shape[0]
    sh = _shape(n, shared)
    lo, hi = 1 / 1.2, 1.2
    sx = torch.rand(sh, device=x.device) * (hi - lo) + lo
    sy = torch.rand(sh, device=x.device) * (hi - lo) + lo
    th = torch.zeros(n, 2, 3, device=x.device)
    th[:, 0, 0], th[:, 1, 1] = sx.expand(n), sy.expand(n)
    return affine_warp(x, th)


def aug_rotate(x, shared):
    n = x.shape[0]
    a = ((torch.rand(_shape(n, shared), device=x.device) - 0.5) * 2 * 15 / 180 * math.pi).expand(n)
    th = torch.zeros(n, 2, 3, device=x.device)
    th[:, 0, 0], th[:, 0, 1], th[:, 1, 0], th[:, 1, 1] = torch.cos(a), -torch.sin(a), torch.sin(a), torch.cos(a)
    return affine_warp(x, th)


def aug_crop(x, shared):
    """Random translation by up to 12.5% of the image size, zero padded."""
    n, _, h, w = x.shape
    r = max(int(h * 0.125 + 0.5) if (h * 0.125) % 1 else int(h * 0.125), 1)
    sh = _shape(n, shared)
    tx = torch.randint(-r, r + 1, sh, device=x.device).expand(n)
    ty = torch.randint(-r, r + 1, sh, device=x.device).expand(n)
    xp = F.pad(x, (r, r, r, r))
    gi, gj = torch.meshgrid(torch.arange(h, device=x.device), torch.arange(w, device=x.device), indexing="ij")
    ii = (gi[None] + r + ty[:, None, None]).clamp(0, h + 2 * r - 1)
    jj = (gj[None] + r + tx[:, None, None]).clamp(0, w + 2 * r - 1)
    return xp[torch.arange(n, device=x.device)[:, None, None], :, ii, jj].permute(0, 3, 1, 2)


def aug_cutout(x, shared):
    n, _, h, w = x.shape
    s = int(h * 0.5 + 0.5)
    sh = _shape(n, shared)
    oy = torch.randint(0, h + (1 - s % 2), sh, device=x.device).expand(n)
    ox = torch.randint(0, w + (1 - s % 2), sh, device=x.device).expand(n)
    gi, gj = torch.meshgrid(torch.arange(h, device=x.device), torch.arange(w, device=x.device), indexing="ij")
    m = ((gi[None] >= (oy - s // 2)[:, None, None]) & (gi[None] < (oy - s // 2 + s)[:, None, None]) &
         (gj[None] >= (ox - s // 2)[:, None, None]) & (gj[None] < (ox - s // 2 + s)[:, None, None]))
    return x * (~m)[:, None].float()


def aug_flip(x, shared):
    n = x.shape[0]
    f = (torch.rand(_shape(n, shared), device=x.device) < 0.5).expand(n)[:, None, None, None]
    return torch.where(f, x.flip(3), x)


def aug_color(x, shared):
    n = x.shape[0]
    sh = (1, 1, 1, 1) if shared else (n, 1, 1, 1)
    x = x + (torch.rand(sh, device=x.device) - 0.5)
    m = x.mean(1, keepdim=True)
    x = (x - m) * (torch.rand(sh, device=x.device) * 2) + m
    m = x.mean((1, 2, 3), keepdim=True)
    return (x - m) * (torch.rand(sh, device=x.device) + 0.5) + m


FAMILIES = {"scale": aug_scale, "rotate": aug_rotate, "crop": aug_crop, "cutout": aug_cutout, "flip": aug_flip, "color": aug_color}


def strategy_for(channels: int) -> str:
    """DSA strategy used by the dataset-condensation benchmarks: colour images use all six families."""
    return "color_crop_cutout_flip_scale_rotate" if channels == 3 else "crop_scale_rotate"


def dsa(x: torch.Tensor, strategy: str, shared: bool = False) -> torch.Tensor:
    """Apply one randomly chosen family of ``strategy`` (underscore separated names) to the batch ``x``."""
    return FAMILIES[random.choice(strategy.split("_"))](x, shared)


def simple_augment(x: torch.Tensor, flip: bool) -> torch.Tensor:
    """Per-sample random translation (+-4 px) and optional horizontal flip."""
    n = x.shape[0]
    th = torch.zeros(n, 2, 3, device=x.device)
    th[:, 0, 0] = 1
    th[:, 1, 1] = 1
    th[:, :, 2] = torch.randint(-4, 5, (n, 2), device=x.device).float() * 2 / x.shape[-1]
    if flip:
        th[:, 0, 0] = torch.where(torch.rand(n, device=x.device) < 0.5, -1.0, 1.0)
    return affine_warp(x, th)
