"""Command line interface: ``coreset-ot select | quality | evaluate | methods``."""
from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

import numpy as np

from ._version import __version__
from .selection import METHODS, CoresetResult, select_coreset


def _add_data_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dataset", required=True, help="mnist | fmnist | cifar10 | cifar100 | tiny-imagenet")
    p.add_argument("--root", default="~/.cache/coreset_ot", help="dataset directory (default: %(default)s)")
    p.add_argument("--download", action="store_true", help="let torchvision download the dataset")
    p.add_argument("--device", default=None, help="cpu | cuda | mps (default: best available)")


def _cmd_methods(_args) -> int:
    for m in METHODS:
        print(m)
    return 0


def _cmd_select(args) -> int:
    from .datasets import load_dataset

    xtr, ytr, _, _ = load_dataset(args.dataset, args.root, args.download)
    res = select_coreset(xtr, ytr, ipc=args.ipc, method=args.method, features=args.features, n_nets=args.n_nets,
                         seed=args.seed, device=args.device, verbose=args.verbose)
    res.save(args.out)
    print(f"selected {len(res)} examples ({args.ipc} per class) -> {args.out}")
    print("class-averaged quality:", json.dumps({k: round(v, 4) for k, v in res.summary().items()}))
    return 0


def _cmd_quality(args) -> int:
    from .datasets import load_dataset
    from .features import as_blocks, build_extractor
    from .metrics import feature_scales, quality_report

    xtr, ytr, _, _ = load_dataset(args.dataset, args.root, args.download)
    res = CoresetResult.load(args.indices)
    ext = build_extractor(args.features, args.n_nets, args.seed, args.device)
    out = {}
    for c in np.unique(res.labels):
        ids = np.where(ytr.numpy() == c)[0]
        blocks = as_blocks(ext(xtr[ids]), args.device)
        pos = {int(g): i for i, g in enumerate(ids)}
        sel = [pos[int(g)] for g in res.indices[res.labels == c]]
        out[int(c)] = quality_report(blocks, sel, feature_scales(blocks, seed=int(c)))
    mean = {k: float(np.mean([v[k] for v in out.values()])) for k in ("w1", "eps", "mmd2")}
    print(json.dumps({"class_averaged": mean}, indent=1))
    return 0


def _cmd_evaluate(args) -> int:
    from .datasets import load_dataset
    from .eval import evaluate_coreset

    xtr, ytr, xte, yte = load_dataset(args.dataset, args.root, args.download)
    res = CoresetResult.load(args.indices)
    out = evaluate_coreset(res, xtr, ytr, xte, yte, arch=args.arch, seeds=range(args.seeds),
                           use_weights=args.weights, aug=args.aug, device=args.device, epochs=args.epochs)
    print(json.dumps({"arch": args.arch, "aug": args.aug, "mean_acc": round(out["mean"] * 100, 2),
                      "std": round(out["std"] * 100, 2), "accs": [round(a * 100, 2) for a in out["accs"]]}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="coreset-ot", description=__doc__)
    ap.add_argument("--version", action="version", version=f"coreset-ot {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("methods", help="list selection methods").set_defaults(func=_cmd_methods)

    s = sub.add_parser("select", help="select a class-balanced coreset and save it as JSON")
    _add_data_args(s)
    s.add_argument("--ipc", type=int, default=10, help="images per class")
    s.add_argument("--method", default="hybrid", choices=METHODS)
    s.add_argument("--features", default="random_convnet")
    s.add_argument("--n-nets", type=int, default=1)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", required=True, help="output JSON file")
    s.add_argument("--verbose", action="store_true")
    s.set_defaults(func=_cmd_select)

    q = sub.add_parser("quality", help="transport/MMD quality of a saved selection in a feature space")
    _add_data_args(q)
    q.add_argument("--indices", required=True)
    q.add_argument("--features", default="random_convnet")
    q.add_argument("--n-nets", type=int, default=8)
    q.add_argument("--seed", type=int, default=999, help="use a seed different from the one used for selection")
    q.set_defaults(func=_cmd_quality)

    e = sub.add_parser("evaluate", help="train a network from scratch on a saved coreset and report test accuracy")
    _add_data_args(e)
    e.add_argument("--indices", required=True)
    e.add_argument("--arch", default="convnet", choices=["convnet", "resnet18", "vgg11", "alexnet"])
    e.add_argument("--aug", default=None, choices=[None, "dsa", "simple"])
    e.add_argument("--seeds", type=int, default=3)
    e.add_argument("--epochs", type=int, default=None)
    e.add_argument("--weights", action="store_true", help="use Voronoi loss weights")
    e.set_defaults(func=_cmd_evaluate)
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
