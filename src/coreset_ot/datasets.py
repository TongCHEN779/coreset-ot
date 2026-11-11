"""Dataset loaders returning normalised float tensors ``(x_train, y_train, x_test, y_test)``.

Requires the ``datasets`` extra (``pip install coreset-ot[datasets]``). Per-channel mean/std are computed on the
training images. Tiny-ImageNet must be unpacked by the user (``tiny-imagenet-200/``)."""
from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import torch

NUM_CLASSES = {"mnist": 10, "fmnist": 10, "cifar10": 10, "cifar100": 100, "tiny-imagenet": 200}
DATASETS = tuple(NUM_CLASSES)


def _need_torchvision():
    try:
        import torchvision  # noqa: F401
        return torchvision
    except ImportError as e:  # pragma: no cover
        raise ImportError("loading datasets requires torchvision: pip install 'coreset-ot[datasets]'") from e


def _normalise(xtr: torch.Tensor, xte: torch.Tensor):
    m = xtr.mean((0, 2, 3), keepdim=True)
    s = xtr.std((0, 2, 3), keepdim=True)
    return (xtr - m) / s, (xte - m) / s


def load_dataset(name: str, root: str = "~/.cache/coreset_ot", download: bool = False) -> Tuple[torch.Tensor, ...]:
    """Load ``name`` from :data:`DATASETS`. With ``download=True`` torchvision downloads the standard datasets."""
    name = name.lower()
    if name not in NUM_CLASSES:
        raise ValueError(f"unknown dataset {name!r}; choose from {DATASETS}")
    root = os.path.expanduser(root)
    if name == "tiny-imagenet":
        return load_tiny_imagenet(root)
    tv = _need_torchvision()
    cls = {"mnist": tv.datasets.MNIST, "fmnist": tv.datasets.FashionMNIST, "cifar10": tv.datasets.CIFAR10,
           "cifar100": tv.datasets.CIFAR100}[name]
    tr, te = cls(root, True, download=download), cls(root, False, download=download)

    def conv(ds):
        x = torch.tensor(np.array(ds.data), dtype=torch.float32) / 255.0
        x = x.unsqueeze(1) if x.ndim == 3 else x.permute(0, 3, 1, 2)
        return x, torch.as_tensor(np.array(ds.targets), dtype=torch.int64)

    (xtr, ytr), (xte, yte) = conv(tr), conv(te)
    xtr, xte = _normalise(xtr, xte)
    return xtr, ytr, xte, yte


def load_tiny_imagenet(root: str) -> Tuple[torch.Tensor, ...]:
    """Tiny-ImageNet-200 from ``root/tiny-imagenet-200``: 100k training and 10k validation images (64x64).

    Memory use is about 5 GB as float32; the validation split is used as the test set."""
    from PIL import Image

    base = os.path.join(root, "tiny-imagenet-200")
    wn = [l.strip() for l in open(os.path.join(base, "wnids.txt"))]
    idx = {w: i for i, w in enumerate(wn)}
    xs, ys = [], []
    for w in wn:
        d = os.path.join(base, "train", w, "images")
        for f in sorted(os.listdir(d)):
            xs.append(np.array(Image.open(os.path.join(d, f)).convert("RGB")))
            ys.append(idx[w])
    xv, yv = [], []
    for line in open(os.path.join(base, "val", "val_annotations.txt")):
        f, w = line.split("\t")[:2]
        xv.append(np.array(Image.open(os.path.join(base, "val", "images", f)).convert("RGB")))
        yv.append(idx[w])
    xtr = torch.tensor(np.stack(xs)).permute(0, 3, 1, 2).float() / 255.0
    xte = torch.tensor(np.stack(xv)).permute(0, 3, 1, 2).float() / 255.0
    xtr, xte = _normalise(xtr, xte)
    return xtr, torch.tensor(ys), xte, torch.tensor(yv)
