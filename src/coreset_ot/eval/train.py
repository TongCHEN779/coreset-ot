"""Train-from-scratch evaluation protocol of the dataset-condensation literature."""
from __future__ import annotations

import math
import random
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ..devices import DeviceLike, get_device
from .augment import dsa, simple_augment, strategy_for
from .models import make_model


def default_epochs(n_train: int, batch_size: int = 256, max_steps: int = 2400, max_epochs: int = 300) -> int:
    """``min(300, ceil(2400 / steps_per_epoch))``: the epoch rule used in the benchmark experiments (keeps large coresets cheap)."""
    return int(min(max_epochs, math.ceil(max_steps / math.ceil(n_train / batch_size))))


def train_and_evaluate(train_x, train_y, test_x, test_y, num_classes: Optional[int] = None, *, arch: str = "convnet",
                       seed: int = 0, epochs: Optional[int] = None, lr: float = 0.01, batch_size: int = 256,
                       weight_decay: float = 5e-4, momentum: float = 0.9, aug: Optional[str] = None,
                       weights=None, device: DeviceLike = None) -> float:
    """Train ``arch`` from scratch on the given examples and return the top-1 accuracy on the test set.

    SGD with momentum, learning rate decayed by 10 at half of the epochs. ``aug`` is ``None``, ``"dsa"`` (the DSA
    strategy of the benchmarks) or ``"simple"`` (translation, plus flips for colour images). ``weights`` are optional
    per-example loss weights (e.g. ``CoresetResult.loss_weights()``)."""
    dev = get_device(device)
    train_x, test_x = torch.as_tensor(train_x).float(), torch.as_tensor(test_x).float()
    train_y, test_y = torch.as_tensor(train_y).long(), torch.as_tensor(test_y).long()
    num_classes = num_classes or int(max(train_y.max(), test_y.max())) + 1
    epochs = epochs or default_epochs(len(train_x), batch_size)
    torch.manual_seed(seed)
    random.seed(seed)
    c, size = train_x.shape[1], train_x.shape[2]
    net = make_model(arch, c, num_classes, size).to(dev)
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [max(epochs // 2, 1)], 0.1)
    X, Y = train_x.to(dev), train_y.to(dev)
    w = None
    if weights is not None:
        w = torch.as_tensor(weights).float().to(dev)
        w = w / w.mean()
    n = len(X)
    strategy = strategy_for(c)
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = X[idx]
            if aug == "dsa":
                xb = dsa(xb, strategy, shared=False)
            elif aug == "simple":
                xb = simple_augment(xb, flip=(c == 3))
            elif aug:
                raise ValueError("aug must be None, 'dsa' or 'simple'")
            loss = F.cross_entropy(net(xb), Y[idx], reduction="none")
            loss = (loss * w[idx]).mean() if w is not None else loss.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    net.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(test_x), 2000):
            correct += (net(test_x[i:i + 2000].to(dev)).argmax(1).cpu() == test_y[i:i + 2000]).sum().item()
    return correct / len(test_x)


def evaluate_coreset(result, train_x, train_y, test_x, test_y, *, arch: str = "convnet", seeds: Sequence[int] = (0, 1, 2),
                     use_weights: bool = False, **kwargs) -> Dict[str, object]:
    """Train on ``result.subset(train_x, train_y)`` for several seeds; returns ``{"accs", "mean", "std"}``."""
    xs, ys = result.subset(train_x, train_y)
    w = result.loss_weights() if use_weights else None
    accs = [train_and_evaluate(xs, ys, test_x, test_y, arch=arch, seed=s, weights=w, **kwargs) for s in seeds]
    return {"accs": accs, "mean": float(np.mean(accs)), "std": float(np.std(accs))}
