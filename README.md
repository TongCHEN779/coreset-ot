# coreset-ot

[![CI](https://github.com/TongCHEN779/coreset-ot/actions/workflows/ci.yml/badge.svg)](https://github.com/TongCHEN779/coreset-ot/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)

**Training-free coreset selection for image classification:** pick a small, class-balanced subset of real training
images without training anything.

It selects in the feature space of *randomly initialised* networks (or any features you supply) with a **k-median**
selector, **kernel herding**, or a **hybrid** of the two, and includes tools to measure how good a selection is.

*Beta (0.1). 88 unit tests run on CPU; the installed package was run end to end on Apple MPS; CUDA is untested.*

## Install

```bash
pip install coreset-ot                 # core library (torch, numpy)
pip install "coreset-ot[datasets]"     # + torchvision/pillow for the dataset loaders and the CLI
```

From source: `pip install -e ".[dev]"`, then `pytest`.

## Quick start

```python
from coreset_ot import select_coreset

# images: float tensor (N, C, H, W), already normalised; labels: integer tensor (N,)
result = select_coreset(images, labels, ipc=10)           # 10 images per class
subset_x, subset_y = result.subset(images, labels)

result.indices      # global indices of the selected images
result.summary()    # class-averaged quality: transport cost (w1), covering radius (eps), mmd2
```

Use your own embeddings instead of random networks:

```python
select_coreset(None, labels, ipc=10, features=my_embeddings)   # (N, d) array
```

Train a network from scratch on the selection to check it (needs a test set):

```python
from coreset_ot.eval import evaluate_coreset
out = evaluate_coreset(result, images, labels, test_x, test_y, arch="convnet", seeds=(0, 1, 2))
print(out["mean"], out["std"])
```

Or from the command line:

```bash
coreset-ot select   --dataset cifar10 --root ~/data --ipc 10 --out sel.json
coreset-ot evaluate --dataset cifar10 --root ~/data --indices sel.json --arch convnet --seeds 3
```

## Which method?

| `method` | Idea | Use it for |
|---|---|---|
| **`hybrid`** (default) | k-median on half the budget, then herding for the rest | most cases |
| `kmedian` | minimise the *mean* distance to the nearest selected image | very small budgets (1-10 per class) |
| `herding` | keep the mean feature of the selection close to the class mean | larger budgets (about 50 per class) |

Baselines for comparison: `kcenter`, `kmedoids`, `facility_location`, `random`. Greedy k-center minimises the
*maximum* distance and is **worse than random** at small budgets; see [methods](docs/methods.md) for why.

## Results at a glance

ConvNet trained from scratch, no augmentation. Accuracy %, mean over 5 seeds (CIFAR-100: 3); 8 random networks.

| Dataset | Images per class | Random | k-median | Herding | Hybrid |
|---|---|---|---|---|---|
| CIFAR-10 | 10 | 27.4 | 36.9 | 38.0 | **38.4** |
| CIFAR-10 | 50 | 42.8 | 48.0 | 51.0 | **51.2** |
| CIFAR-100 | 10 | 14.4 | **21.2** | 20.4 | 21.0 |
| Fashion-MNIST | 10 | 74.5 | 79.3 | 79.1 | **80.3** |

All budgets and datasets: [docs/benchmarks.md](docs/benchmarks.md). Reproduce the CIFAR-10 row with
`examples/reproduce_cifar10.py` (38.4 +- 0.7 % with this package on Apple MPS).

## Good to know

* **Architecture matters.** With a ConvNet, random-network features match the features of a *trained* network at
  about 1/30 of the selection cost. For ResNet-18, VGG-11 and AlexNet, a trained network's features gave better
  coresets in our experiments.
* **Selection is not synthesis.** Selected real images are interpretable and need no optimisation, but methods that
  synthesise images (e.g. distribution matching) were more accurate at small budgets.
* **Scale.** Distance matrices are computed per class (`n x n`); classes above 12,000 images are subsampled with a
  warning (herding is unaffected).

More in [Limitations](docs/limitations.md) and the [FAQ](docs/faq.md).

## Documentation

[Methods](docs/methods.md) · [API](docs/api.md) · [Benchmarks](docs/benchmarks.md) ·
[Reproducing the numbers](docs/reproducing.md) · [Limitations](docs/limitations.md) · [FAQ](docs/faq.md)

## Citation

Please cite this software using [`CITATION.cff`](CITATION.cff) (GitHub's "Cite this repository" button).

## License

MIT. See [LICENSE](LICENSE).
