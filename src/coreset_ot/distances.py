"""Pairwise distances and the ensemble metric used by the selectors."""
from __future__ import annotations

from typing import Optional, Sequence

import torch

_MAX_MEDIAN_ELEMS = 16_000_000


def pairwise_distances(a: torch.Tensor, b: Optional[torch.Tensor] = None, chunk: int = 4096) -> torch.Tensor:
    """Euclidean distance matrix between rows of ``a`` and ``b`` (``b=a`` if omitted), computed in row chunks."""
    a = a.float()
    b = a if b is None else b.float()
    if a.shape[0] <= chunk:
        return torch.cdist(a, b)
    return torch.cat([torch.cdist(a[i:i + chunk], b) for i in range(0, a.shape[0], chunk)])


def median_scale(D: torch.Tensor, seed: int = 0) -> float:
    """Median of the entries of ``D`` (the diagonal zeros included, as in the reference experiments).

    For very large matrices the median is estimated on a random sample of entries."""
    flat = D.reshape(-1)
    if flat.numel() > _MAX_MEDIAN_ELEMS:
        g = torch.Generator().manual_seed(seed)
        idx = torch.randint(0, flat.numel(), (_MAX_MEDIAN_ELEMS,), generator=g).to(D.device)
        flat = flat[idx]
    return float(flat.median())


def ensemble_distance(blocks: Sequence[torch.Tensor]) -> torch.Tensor:
    """Root-mean-square over feature blocks of the median-normalised pairwise distances.

    ``blocks`` is a list of ``(n, d_k)`` tensors, e.g. the features of ``K`` random networks. For a single
    block this is the Euclidean distance divided by its median."""
    acc = None
    for f in blocks:
        Dk = pairwise_distances(f)
        s = max(median_scale(Dk), 1e-12)
        Dk = (Dk / s) ** 2
        acc = Dk if acc is None else acc + Dk
    return (acc / len(blocks)).sqrt()


def concat_normalised(blocks: Sequence[torch.Tensor]) -> torch.Tensor:
    """Concatenate blocks after dividing each by (mean row norm * sqrt(K)); used as the herding feature map."""
    K = len(blocks)
    return torch.cat([f.float() / (f.float().norm(dim=1).mean().clamp_min(1e-12) * K ** 0.5) for f in blocks], dim=1)
