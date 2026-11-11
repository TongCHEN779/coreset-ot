"""Feature extractors: randomly initialised networks (no training), pixels, or anything you provide.

An extractor is a callable ``x -> list of (n, d_k) tensors`` (a list of *feature blocks*). Selectors average the
median-normalised distances over blocks, so several random networks (or several feature spaces) can be combined.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Union

import numpy as np
import torch

from .devices import DeviceLike, get_device
from .eval.models import ARCHS, make_model, uses_batchnorm

FeatureBlocks = List[torch.Tensor]
Extractor = Callable[[torch.Tensor], FeatureBlocks]


def _fresh(seed: int, build):
    """Build a module under a fixed seed without disturbing the global RNG state."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return build()


class RandomNetFeatures:
    """Penultimate features of ``n_nets`` randomly initialised networks of one architecture.

    Parameters
    ----------
    arch: one of ``coreset_ot.eval.ARCHS``.
    n_nets: number of independent initialisations (blocks). One is usually enough.
    seed: base seed; network ``k`` is initialised with ``7919 * seed + 104729 + k``.
    batch_size: images per forward pass. For networks with batch normalisation (ResNet-18, VGG-11) batch
        statistics are used, so features depend mildly on the batch composition.
    """

    def __init__(self, arch: str = "convnet", n_nets: int = 1, seed: int = 0, device: DeviceLike = None,
                 batch_size: int = 1000, bn_batch_stats: Optional[bool] = None):
        if arch not in ARCHS:
            raise ValueError(f"unknown architecture {arch!r}; choose from {ARCHS}")
        self.arch, self.n_nets, self.seed, self.batch_size = arch, n_nets, seed, batch_size
        self.device = get_device(device)
        self.bn_batch_stats = uses_batchnorm(arch) if bn_batch_stats is None else bn_batch_stats
        self._nets: Optional[list] = None
        self._shape = None

    def _build(self, x: torch.Tensor) -> None:
        c, h = x.shape[1], x.shape[2]
        self._shape = (c, h)
        self._nets = [_fresh(7919 * self.seed + 104729 + k, lambda: make_model(self.arch, c, 2, h)).to(self.device)
                      for k in range(self.n_nets)]

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> FeatureBlocks:
        if x.ndim != 4:
            raise ValueError("expected images of shape (N, C, H, W)")
        if self._nets is None:
            self._build(x)
        elif (x.shape[1], x.shape[2]) != self._shape:
            raise ValueError(f"extractor was built for (C, H) = {self._shape}, got {(x.shape[1], x.shape[2])}")
        blocks = []
        for net in self._nets:
            net.train(self.bn_batch_stats)
            parts = [net.embed(x[i:i + self.batch_size].to(self.device).float()).float() for i in range(0, len(x), self.batch_size)]
            blocks.append(torch.cat(parts))
        return blocks


class MixedArchFeatures:
    """One randomly initialised network of each of several architectures (ConvNet, ResNet-18, VGG-11, AlexNet).

    The mixture makes the selection less tied to the inductive bias of a single architecture; in our experiments
    it helped for ResNet-18 and VGG-11 but not consistently (see the documentation)."""

    def __init__(self, archs: Sequence[str] = ARCHS, seed: int = 0, device: DeviceLike = None, batch_size: int = 1000):
        self.archs, self.seed, self.batch_size = tuple(archs), seed, batch_size
        self.device = get_device(device)
        self._nets: Optional[list] = None

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> FeatureBlocks:
        if self._nets is None:
            c, h = x.shape[1], x.shape[2]
            self._nets = [(a, _fresh(31 * self.seed + 1 + i, lambda a=a: make_model(a, c, 2, h)).to(self.device))
                          for i, a in enumerate(self.archs)]
        blocks = []
        for arch, net in self._nets:
            net.train(uses_batchnorm(arch))
            parts = [net.embed(x[i:i + self.batch_size].to(self.device).float()).float() for i in range(0, len(x), self.batch_size)]
            blocks.append(torch.cat(parts))
        return blocks


class PixelFeatures:
    """Flattened pixels (a weak baseline metric)."""

    def __init__(self, device: DeviceLike = None):
        self.device = get_device(device)

    def __call__(self, x: torch.Tensor) -> FeatureBlocks:
        return [x.reshape(len(x), -1).to(self.device).float()]


def as_blocks(out: Union[torch.Tensor, np.ndarray, Sequence], device: DeviceLike = None) -> FeatureBlocks:
    """Normalise extractor output (array, tensor or list of them) to a list of float tensors."""
    dev = get_device(device)
    if isinstance(out, (torch.Tensor, np.ndarray)):
        out = [out]
    blocks = [torch.as_tensor(b).float().to(dev) for b in out]
    if not blocks or any(b.ndim != 2 or b.shape[0] != blocks[0].shape[0] for b in blocks):
        raise ValueError("features must be (n, d) arrays with the same number of rows")
    return blocks


_NAMED = {
    "random_convnet": lambda n, s, d: RandomNetFeatures("convnet", n, s, d),
    "random_resnet18": lambda n, s, d: RandomNetFeatures("resnet18", n, s, d),
    "random_vgg11": lambda n, s, d: RandomNetFeatures("vgg11", n, s, d),
    "random_alexnet": lambda n, s, d: RandomNetFeatures("alexnet", n, s, d),
    "random_mixed": lambda n, s, d: MixedArchFeatures(seed=s, device=d),
    "pixels": lambda n, s, d: PixelFeatures(d),
}
FEATURE_NAMES = tuple(_NAMED)


def build_extractor(features, n_nets: int = 1, seed: int = 0, device: DeviceLike = None):
    """Resolve the ``features`` argument of :func:`coreset_ot.select_coreset` to a callable extractor."""
    if isinstance(features, str):
        if features not in _NAMED:
            raise ValueError(f"unknown features {features!r}; choose from {FEATURE_NAMES} or pass an extractor/array")
        return _NAMED[features](n_nets, seed, device)
    if callable(features):
        return features
    raise TypeError("features must be a name, a callable extractor, or precomputed arrays")
