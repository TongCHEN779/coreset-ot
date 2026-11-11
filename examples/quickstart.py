"""Select 10 images per class from CIFAR-10 and report the transport cost, then train a ConvNet on them."""
import sys

from coreset_ot import select_coreset
from coreset_ot.datasets import load_dataset
from coreset_ot.eval import evaluate_coreset

root = sys.argv[1] if len(sys.argv) > 1 else "~/.cache/coreset_ot"
xtr, ytr, xte, yte = load_dataset("cifar10", root, download=True)

result = select_coreset(xtr, ytr, ipc=10, method="hybrid", seed=0, verbose=False)
print("selected", len(result), "images; class-averaged quality:", {k: round(v, 3) for k, v in result.summary().items()})

rand = select_coreset(xtr, ytr, ipc=10, method="random", seed=0)
print("random selection quality:", {k: round(v, 3) for k, v in rand.summary().items()})

for name, r in [("hybrid", result), ("random", rand)]:
    out = evaluate_coreset(r, xtr, ytr, xte, yte, arch="convnet", seeds=(0, 1, 2))
    print(f"{name:7s} ConvNet accuracy: {100 * out['mean']:.1f} +- {100 * out['std']:.1f}")
