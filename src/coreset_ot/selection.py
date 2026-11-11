"""High-level class-balanced selection API."""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch

from . import solvers
from .devices import DeviceLike, empty_cache, get_device
from .distances import concat_normalised, ensemble_distance
from .features import as_blocks, build_extractor
from .metrics import feature_scales, quality_report, voronoi_weights

METHODS = ("hybrid", "kmedian", "herding", "kcenter", "kmedoids", "facility_location", "random")
_NEEDS_DISTANCE = ("hybrid", "kmedian", "kcenter", "kmedoids", "facility_location")


@dataclass
class CoresetResult:
    """Outcome of :func:`select_coreset`.

    Attributes
    ----------
    indices: global row indices of the selected examples (grouped by class, in selection order).
    labels: their class labels.
    weights: Voronoi weights, i.e. the fraction of its class nearest to each selected example (sums to one per class).
    per_class: ``{class: {"w1", "eps", "mmd2", "n"}}`` quality measures in the selection feature space.
    """

    indices: np.ndarray
    labels: np.ndarray
    weights: np.ndarray
    method: str
    ipc: int
    params: Dict = field(default_factory=dict)
    per_class: Dict[int, Dict[str, float]] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.indices)

    def loss_weights(self) -> np.ndarray:
        """Per-example loss weights: Voronoi weights rescaled so that every class has total weight ``ipc``."""
        out = np.zeros_like(self.weights, dtype=np.float64)
        for c in np.unique(self.labels):
            m = self.labels == c
            w = self.weights[m].astype(np.float64)
            out[m] = w / max(w.sum(), 1e-12) * m.sum()
        return out

    def summary(self) -> Dict[str, float]:
        """Class-averaged quality measures."""
        if not self.per_class:
            return {}
        keys = ("w1", "eps", "mmd2")
        return {k: float(np.mean([v[k] for v in self.per_class.values() if k in v])) for k in keys}

    def subset(self, data, labels=None):
        """``data[indices]`` (and the matching labels if ``labels`` is given)."""
        if labels is None:
            return data[self.indices]
        return data[self.indices], labels[self.indices]

    def to_dict(self) -> Dict:
        return {"indices": self.indices.tolist(), "labels": self.labels.tolist(), "weights": self.weights.tolist(),
                "method": self.method, "ipc": self.ipc, "params": self.params,
                "per_class": {str(k): v for k, v in self.per_class.items()}}

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str) -> "CoresetResult":
        with open(path) as f:
            d = json.load(f)
        return cls(np.array(d["indices"]), np.array(d["labels"]), np.array(d["weights"]), d["method"], d["ipc"],
                   d.get("params", {}), {int(k): v for k, v in d.get("per_class", {}).items()})


def _select_class(blocks, m: int, method: str, rng_seed: int, fraction: float, max_exact: int) -> List[int]:
    n = blocks[0].shape[0]
    if m >= n:
        return list(range(n))
    if method == "random":
        return [int(i) for i in np.random.default_rng(rng_seed).choice(n, m, replace=False)]
    if method == "herding":
        return solvers.herding(concat_normalised(blocks), m)
    pool = None
    if n > max_exact:                                      # distance matrices are n x n: work on a random pool
        warnings.warn(f"class of size {n} exceeds max_exact={max_exact}; selecting from a random pool of that size",
                      RuntimeWarning, stacklevel=3)
        pool = torch.as_tensor(np.sort(np.random.default_rng(rng_seed).choice(n, max_exact, replace=False)), device=blocks[0].device)
        blocks = [b[pool] for b in blocks]
    D = ensemble_distance(blocks)
    if method == "kmedian":
        sel = solvers.kmedian(D, m)
    elif method == "facility_location":
        sel = solvers.facility_location_greedy(D, m)
    elif method == "kcenter":
        sel = solvers.kcenter_greedy(D, m, seed=rng_seed)
    elif method == "kmedoids":
        sel = solvers.kmedoids_alternating(D, m, seed=rng_seed)
    elif method == "hybrid":
        sel = solvers.hybrid(D, concat_normalised(blocks), m, fraction=fraction)
    else:
        raise ValueError(method)
    if pool is not None:
        sel = [int(pool[i]) for i in sel]
    return sel


def select_coreset(data=None, labels=None, ipc: int = 10, *, method: str = "hybrid", features="random_convnet",
                   n_nets: int = 1, seed: int = 0, device: DeviceLike = None, hybrid_fraction: float = 0.5,
                   max_exact: int = 12000, quality: bool = True, verbose: bool = False) -> CoresetResult:
    """Select ``ipc`` examples per class.

    Parameters
    ----------
    data: images ``(N, C, H, W)`` (tensor or array, normalised) - only needed when ``features`` is a name or an
        extractor. Not needed when ``features`` is a precomputed ``(N, d)`` array (or a list of such arrays).
    labels: ``(N,)`` integer class labels.
    ipc: images (examples) per class. Classes with fewer than ``ipc`` examples are taken whole.
    method: ``"hybrid"`` (default), ``"kmedian"``, ``"herding"``, ``"kcenter"``, ``"kmedoids"``,
        ``"facility_location"`` (greedy only) or ``"random"``.
    features: ``"random_convnet"`` (default), ``"random_resnet18"``, ``"random_vgg11"``, ``"random_alexnet"``,
        ``"random_mixed"``, ``"pixels"``; a callable ``x -> (n, d) array`` (or list of arrays); or precomputed features.
    n_nets: random networks (feature blocks) to average when ``features`` is a ``"random_*"`` name.
    seed: seed for the random networks and the randomised solvers.
    hybrid_fraction: share of the budget given to ``k``-median before herding takes over (hybrid only).
    max_exact: largest class size for which exact ``n x n`` distance matrices are used; larger classes are subsampled.
    quality: compute per-class transport/MMD measures and Voronoi weights.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    if labels is None:
        raise ValueError("labels are required")
    labels = np.asarray(labels.cpu() if isinstance(labels, torch.Tensor) else labels).astype(np.int64)
    dev = get_device(device)

    need_blocks = quality or method != "random"        # random selection without quality needs no features at all
    precomputed = None
    extractor = None
    if isinstance(features, (np.ndarray, torch.Tensor)) or (isinstance(features, (list, tuple)) and features and
                                                              isinstance(features[0], (np.ndarray, torch.Tensor))):
        precomputed = as_blocks(features, "cpu")
        if precomputed[0].shape[0] != len(labels):
            raise ValueError("precomputed features and labels must have the same number of rows")
    else:
        if need_blocks:
            if data is None:
                raise ValueError("`data` is required unless `features` is precomputed")
            extractor = build_extractor(features, n_nets, seed, dev)
            if not isinstance(data, torch.Tensor):
                data = torch.as_tensor(np.asarray(data))

    idx_all, lab_all, w_all, per_class = [], [], [], {}
    for c in np.unique(labels):
        ids = np.where(labels == c)[0]
        rng_seed = seed * 10007 + 31 * int(c) + ipc
        if precomputed is not None:
            blocks = [b[torch.as_tensor(ids)].to(dev) for b in precomputed]
        elif need_blocks:
            blocks = as_blocks(extractor(data[torch.as_tensor(ids)]), dev)
        else:
            blocks = [torch.zeros(len(ids), 1)]
        m = min(ipc, len(ids))
        sel = _select_class(blocks, m, method, rng_seed, hybrid_fraction, max_exact)
        if quality:
            sc = feature_scales(blocks, seed=int(c))
            per_class[int(c)] = dict(quality_report(blocks, sel, sc), n=len(ids))
            w = voronoi_weights(blocks, sel, sc).cpu().numpy()
        else:
            w = np.full(len(sel), 1.0 / len(sel))
        idx_all.append(ids[np.asarray(sel)])
        lab_all.append(np.full(len(sel), c))
        w_all.append(w)
        if verbose:
            print(f"class {int(c)}: selected {len(sel)} of {len(ids)}", flush=True)
        del blocks
        empty_cache(dev)
    return CoresetResult(np.concatenate(idx_all), np.concatenate(lab_all), np.concatenate(w_all), method, ipc,
                         dict(features=features if isinstance(features, str) else "custom", n_nets=n_nets, seed=seed,
                              hybrid_fraction=hybrid_fraction), per_class)
