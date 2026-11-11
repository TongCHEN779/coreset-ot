"""Device selection helpers."""
from __future__ import annotations

from typing import Union

import torch

DeviceLike = Union[None, str, torch.device]


def get_device(device: DeviceLike = None) -> torch.device:
    """Return ``device`` as a :class:`torch.device`; ``None`` picks CUDA, then Apple MPS, then CPU."""
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def empty_cache(device: torch.device) -> None:
    """Release cached accelerator memory (no-op on CPU)."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
