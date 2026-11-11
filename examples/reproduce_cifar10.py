"""Reproduce the documented CIFAR-10 hybrid result: 8 random ConvNets, ConvNet trained from scratch, 5 seeds.
Expected (original runs): 38.4 +- 0.3 % at 10 images per class, 51.2 +- 0.8 % at 50 (same protocol, Apple MPS)."""
import sys

import numpy as np

from coreset_ot import select_coreset
from coreset_ot.datasets import load_dataset
from coreset_ot.eval import evaluate_coreset

root = sys.argv[1] if len(sys.argv) > 1 else "~/.cache/coreset_ot"
ipc = int(sys.argv[2]) if len(sys.argv) > 2 else 10
xtr, ytr, xte, yte = load_dataset("cifar10", root, download=True)

accs = []
for seed in range(5):                      # the seed changes both the selection and the network initialisation
    r = select_coreset(xtr, ytr, ipc=ipc, method="hybrid", features="random_convnet", n_nets=8, seed=seed)
    accs += evaluate_coreset(r, xtr, ytr, xte, yte, arch="convnet", seeds=(seed,))["accs"]
    print(f"seed {seed}: {100 * accs[-1]:.1f}", flush=True)
print(f"hybrid, {ipc} per class: {100 * np.mean(accs):.1f} +- {100 * np.std(accs):.1f}")
