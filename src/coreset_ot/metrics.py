"""Quality measures of a selection: transport cost, covering radius and MMD, measured in a feature space.

For a class with feature blocks ``F_1..F_K`` and selected rows ``S``, the distance between rows ``i`` and ``j`` is
``d(i, j) = sqrt(mean_k (|F_k[i] - F_k[j]| / s_k)^2)`` where ``s_k`` is the median pairwise distance of block ``k``
(estimated on random pairs). Then

* ``w1``  - mean distance to the nearest selected row: the 1-Wasserstein distance between the data and the selected
  rows carrying their Voronoi weights (the tightest transport bound on the loss gap of a weighted coreset);
* ``eps`` - maximum distance to the nearest selected row (covering radius);
* ``mmd2``- squared distance between the mean feature of the data and of the selection, averaged over blocks and
  normalised by the mean squared feature norm.

Measuring quality in features of *other* random networks than those used for selection gives an unbiased check
(see the documentation).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch


def feature_scales(blocks: Sequence[torch.Tensor], n_pairs: int = 2000, seed: int = 0) -> List[float]:
    """Median pairwise distance of each block, estimated on ``n_pairs`` random pairs."""
    n = blocks[0].shape[0]
    g = torch.Generator().manual_seed(seed)
    a = torch.randint(0, n, (n_pairs,), generator=g)
    b = torch.randint(0, n, (n_pairs,), generator=g)
    out = []
    for f in blocks:
        d = (f[a.to(f.device)] - f[b.to(f.device)]).norm(dim=1)
        out.append(max(float(d.median()), 1e-12))
    return out


def distances_to(blocks: Sequence[torch.Tensor], idx: Sequence[int], scales: Optional[Sequence[float]] = None) -> torch.Tensor:
    """``(n, m)`` ensemble distances from every row to the selected rows ``idx``."""
    scales = feature_scales(blocks) if scales is None else scales
    sel = torch.as_tensor(list(idx), device=blocks[0].device)
    acc = 0
    for f, s in zip(blocks, scales):
        acc = acc + (torch.cdist(f, f[sel], compute_mode="donot_use_mm_for_euclid_dist") / s) ** 2   # exact: self-distances are 0
    return (acc / len(blocks)).sqrt()


def voronoi_weights(blocks: Sequence[torch.Tensor], idx: Sequence[int], scales: Optional[Sequence[float]] = None) -> torch.Tensor:
    """Fraction of the data nearest to each selected row (sums to one)."""
    D = distances_to(blocks, idx, scales)
    counts = torch.bincount(D.argmin(1), minlength=D.shape[1]).float()
    return counts / counts.sum()


def mean_assignment_distance(blocks, idx, scales=None) -> float:
    """``w1``: mean distance to the nearest selected row."""
    return float(distances_to(blocks, idx, scales).min(1).values.mean())


def covering_radius(blocks, idx, scales=None) -> float:
    """``eps``: maximum distance to the nearest selected row."""
    return float(distances_to(blocks, idx, scales).min(1).values.max())


def mmd2(blocks: Sequence[torch.Tensor], idx: Sequence[int]) -> float:
    """Normalised squared mean-feature discrepancy with equal weights on the selected rows."""
    sel = torch.as_tensor(list(idx), device=blocks[0].device)
    tot = 0.0
    for f in blocks:
        c = float(f.norm(dim=1).mean()) ** 2
        tot += float(((f.mean(0) - f[sel].mean(0)) ** 2).sum()) / max(c, 1e-12)
    return tot / len(blocks)


def quality_report(blocks: Sequence[torch.Tensor], idx: Sequence[int], scales: Optional[Sequence[float]] = None) -> Dict[str, float]:
    """``{"w1", "eps", "mmd2"}`` of a selection within one class."""
    scales = feature_scales(blocks) if scales is None else scales
    D = distances_to(blocks, idx, scales)
    near = D.min(1).values
    return {"w1": float(near.mean()), "eps": float(near.max()), "mmd2": mmd2(blocks, idx)}
