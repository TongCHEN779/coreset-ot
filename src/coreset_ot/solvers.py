"""Selection algorithms on a precomputed (symmetric) distance matrix ``D`` or a feature map ``phi``.

All functions return plain Python lists of row indices. ``D`` is an ``(n, n)`` float tensor of pairwise
distances; ``phi`` is an ``(n, p)`` feature matrix.

Objectives
----------
* :func:`kcenter_greedy`  - minimise the *maximum* distance to the nearest center (covering radius).
* :func:`facility_location_greedy`, :func:`pam_swap`, :func:`kmedian` - minimise the *mean* (optionally
  weighted) distance to the nearest center, i.e. the k-median / k-medoids objective; this equals the
  1-Wasserstein distance between the data and the weighted centers when the weights are the Voronoi cell sizes.
* :func:`herding` - kernel herding on a feature map: match the mean feature vector of the data.
* :func:`hybrid`  - ``k``-median on part of the budget, then herding for the rest.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import numpy as np
import torch


def _check(n: int, m: int) -> None:
    if m < 1:
        raise ValueError("budget must be at least 1")
    if m > n:
        raise ValueError(f"budget {m} exceeds the number of points {n}")


def _weights(D: torch.Tensor, weights: Optional[torch.Tensor]) -> torch.Tensor:
    if weights is None:
        return torch.ones(D.shape[0], device=D.device, dtype=D.dtype)
    return weights.to(device=D.device, dtype=D.dtype)


def kcenter_greedy(D: torch.Tensor, m: int, seed: int = 0) -> List[int]:
    """Greedy farthest-point traversal (Gonzalez, 1985) from a random first center."""
    n = D.shape[0]
    _check(n, m)
    rng = np.random.default_rng(seed)
    first = int(rng.integers(n))
    sel = [first]
    cost = D[first].clone()
    for _ in range(m - 1):
        nxt = int(torch.argmax(cost))
        sel.append(nxt)
        cost = torch.minimum(cost, D[nxt])
    return sel


def facility_location_greedy(D: torch.Tensor, m: int, weights: Optional[torch.Tensor] = None) -> List[int]:
    """Greedy (weighted) facility location with similarity ``S = max(D) - D``.

    Maximising ``sum_j w_j max_i S_ij`` over the chosen centers ``i`` is the same problem as minimising the
    weighted mean distance to the nearest center (k-median), up to a constant. The greedy rule has the usual
    ``1 - 1/e`` guarantee for the facility-location objective; the marginal gain of candidate ``i`` is
    ``sum_j w_j * relu(S_ij - cur_j)`` where ``cur_j`` is the similarity of point ``j`` to its best center so far.
    """
    n = D.shape[0]
    _check(n, m)
    w = _weights(D, weights)
    S = D.max() - D
    cur = torch.zeros(n, device=D.device, dtype=D.dtype)
    avail = torch.ones(n, dtype=torch.bool, device=D.device)
    sel: List[int] = []
    for _ in range(m):
        gain = (w[None, :] * torch.relu(S - cur[None, :])).sum(1)
        gain[~avail] = -1.0
        j = int(gain.argmax())
        sel.append(j)
        avail[j] = False
        cur = torch.maximum(cur, S[j])
    return sel


def kmedian_cost(D: torch.Tensor, sel: Sequence[int], weights: Optional[torch.Tensor] = None) -> float:
    """Weighted *sum* of the distances to the nearest selected point."""
    w = _weights(D, weights)
    return float((w * D[:, list(sel)].min(1).values).sum())


def pam_swap(D: torch.Tensor, selected: Iterable[int], weights: Optional[torch.Tensor] = None,
             max_passes: int = 5, tol: float = 1e-6) -> List[int]:
    """Swap refinement (PAM, Kaufman & Rousseeuw 1990): replace one center at a time by the best data point.

    Never increases the (weighted) k-median cost. Stops after ``max_passes`` passes or when no swap improves
    the cost by more than a relative ``tol``."""
    n = D.shape[0]
    sel = list(selected)
    w = _weights(D, weights)
    best = kmedian_cost(D, sel, w)
    for _ in range(max_passes):
        improved = False
        for pos in range(len(sel)):
            others = [s for k, s in enumerate(sel) if k != pos]
            dm = D[:, others].min(1).values if others else torch.full((n,), 1e9, device=D.device, dtype=D.dtype)
            cand = (w[:, None] * torch.minimum(dm[:, None], D)).sum(0)   # cost if center `pos` became each candidate
            j = int(cand.argmin())
            c = float(cand[j])
            if c < best - tol * abs(best):
                sel[pos] = j
                best = c
                improved = True
        if not improved:
            break
    return sel


def kmedian(D: torch.Tensor, m: int, weights: Optional[torch.Tensor] = None, max_passes: int = 5) -> List[int]:
    """Greedy facility location followed by PAM swaps (the default k-median solver)."""
    return pam_swap(D, facility_location_greedy(D, m, weights), weights, max_passes=max_passes)


def kmedoids_alternating(D: torch.Tensor, m: int, seed: int = 0, iters: int = 30) -> List[int]:
    """Voronoi-iteration k-medoids (Park & Jun 2009) with k-means++ style ``D^1`` seeding."""
    n = D.shape[0]
    _check(n, m)
    rng = np.random.default_rng(seed)
    med = [int(rng.integers(n))]
    dmin = D[:, med[0]].clone()
    for _ in range(m - 1):
        p = (dmin / dmin.sum().clamp_min(1e-12)).cpu().numpy().astype(np.float64)
        p /= p.sum()
        c = int(rng.choice(n, p=p))
        med.append(c)
        dmin = torch.minimum(dmin, D[:, c])
    med_t = torch.tensor(med, device=D.device)
    for _ in range(iters):
        assign = D[:, med_t].argmin(1)
        new = med_t.clone()
        for j in range(m):
            mem = (assign == j).nonzero().squeeze(1)
            if len(mem) == 0:
                continue
            new[j] = mem[D[mem][:, mem].sum(0).argmin()]
        if torch.equal(new, med_t):
            break
        med_t = new
    return med_t.cpu().tolist()


def herding(phi: torch.Tensor, m: int, initial: Sequence[int] = ()) -> List[int]:
    """Kernel herding with the linear kernel on ``phi`` (Welling 2009; Chen et al. 2010).

    Greedily picks the point maximising ``<(t+1) mu - sum_{selected} phi_j, phi_i>`` with ``mu`` the mean
    feature vector, so the mean of the selected features tracks ``mu``. ``initial`` points are kept and the
    procedure continues from them."""
    n = phi.shape[0]
    _check(n, m)
    phi = phi.float()
    mean = phi.mean(0)
    sel = [int(i) for i in initial]
    cur = phi[sel].sum(0) if sel else torch.zeros_like(mean)
    avail = torch.ones(n, dtype=torch.bool, device=phi.device)
    if sel:
        avail[sel] = False
    while len(sel) < m:
        score = ((len(sel) + 1) * mean - cur) @ phi.T
        score[~avail] = -float("inf")
        j = int(score.argmax())
        sel.append(j)
        avail[j] = False
        cur = cur + phi[j]
    return sel


def hybrid(D: torch.Tensor, phi: torch.Tensor, m: int, fraction: float = 0.5, max_passes: int = 5) -> List[int]:
    """``k``-median on ``max(1, round(fraction * m))`` points, then herding continues to ``m`` points."""
    n = D.shape[0]
    _check(n, m)
    m1 = min(m, max(1, int(round(fraction * m))))
    return herding(phi, m, initial=kmedian(D, m1, max_passes=max_passes))
