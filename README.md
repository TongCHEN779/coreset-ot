# coreset-ot

Training-free **coreset selection** (choosing a small, class-balanced subset of real training examples) built on an
optimal-transport view of the problem.

* **No training needed.** Selection runs in the feature space of *randomly initialised* networks (or any features you
  supply): a **k-median** selector, **kernel herding**, and a **hybrid** of the two.
* **Principled objective.** For a weighted subset, the training-loss gap of any Lipschitz loss is bounded by a
  1-Wasserstein distance; its tightest form is the *mean* distance to the nearest selected point (the k-median
  objective), not the covering radius that greedy k-center minimises.
* **Diagnostics.** Transport cost, covering radius and MMD of a selection, so you can compare selections without
  training anything.
* **Evaluation harness.** The train-from-scratch protocol of the dataset-condensation literature (ConvNet, ResNet-18,
  VGG-11, AlexNet; DSA augmentation) to check how good a coreset is.

> **Status: beta (0.1).** 88 unit tests run on CPU. The installed package was run end to end on Apple MPS and
> reproduces the CIFAR-10 hybrid result of the authors' original experiments (38.4 +- 0.7 % at 10 images per class,
> 5 seeds; 38.4 +- 0.3 % in the original runs). CUDA is expected to work but has not been validated by the authors. See
> [Limitations](docs/limitations.md).

## Install

```bash
pip install coreset-ot                 # core library (torch, numpy)
pip install "coreset-ot[datasets]"     # + torchvision/pillow for the dataset loaders and CLI
```

From source: `pip install -e ".[dev]"` and `pytest`.

## Quick start

```python
from coreset_ot import select_coreset

# images: float tensor (N, C, H, W), already normalised; labels: integer tensor (N,)
result = select_coreset(images, labels, ipc=10)            # 10 images per class, hybrid selector
subset_x, subset_y = result.subset(images, labels)

result.indices          # global indices of the selected images
result.weights          # Voronoi weights (fraction of its class nearest to each selected image)
result.summary()        # class-averaged transport cost (w1), covering radius (eps) and mmd2
result.save("coreset.json")
```

Choose the method and the feature space:

```python
select_coreset(images, labels, ipc=50, method="herding")                       # large budgets
select_coreset(images, labels, ipc=1,  method="kmedian")                       # tiny budgets
select_coreset(images, labels, ipc=10, features="random_mixed")                # ConvNet+ResNet+VGG+AlexNet features
select_coreset(None, labels, ipc=10, features=my_embeddings)                   # your own (N, d) features
```

Evaluate by training from scratch (needs the test set):

```python
from coreset_ot.eval import evaluate_coreset
out = evaluate_coreset(result, images, labels, test_x, test_y, arch="convnet", seeds=(0, 1, 2), aug="dsa")
print(out["mean"], out["std"])
```

Command line:

```bash
coreset-ot select --dataset cifar10 --root ~/data --ipc 10 --out cifar10_ipc10.json
coreset-ot quality  --dataset cifar10 --root ~/data --indices cifar10_ipc10.json
coreset-ot evaluate --dataset cifar10 --root ~/data --indices cifar10_ipc10.json --arch convnet --seeds 3
```

## Methods

| `method` | What it does | Use it when |
|---|---|---|
| `hybrid` (default) | k-median on half the budget, then herding for the rest | default; best or near-best at all budgets in our ConvNet experiments |
| `kmedian` | greedy facility location + PAM swaps; minimises the mean distance to the nearest selected point | very small budgets (1-10 per class) |
| `herding` | kernel herding on the feature map; matches the class mean | larger budgets (about 50 per class) |
| `kcenter` | greedy farthest-point; minimises the *maximum* distance | baseline: **worse than random** at small budgets |
| `kmedoids` | alternating k-medoids (k-means++ seeding) | baseline |
| `facility_location` | greedy facility location only (no swaps) | baseline |
| `random` | uniform per class | baseline |

## What to expect (from the authors' experiments)

ConvNet trained from scratch, no augmentation, accuracy %, 1 / 10 / 50 images per class (mean over 5 seeds,
3 for CIFAR-100). Random-network features, 8 networks.

| Dataset | Random | k-median | Herding | Hybrid |
|---|---|---|---|---|
| CIFAR-10 | 14.7 / 27.4 / 42.8 | 20.4 / 36.9 / 48.0 | 22.5 / 38.0 / 51.0 | 20.4 / **38.4** / **51.2** |
| CIFAR-100 | 4.0 / 14.4 / 29.3 | 7.7 / **21.2** / 32.7 | 8.9 / 20.4 / 34.2 | 7.7 / 21.0 / **34.4** |
| FMNIST | 50.3 / 74.5 / 81.9 | 68.4 / 79.3 / 85.0 | 66.8 / 79.1 / 84.9 | 68.4 / **80.3** / **85.3** |

Honest caveats (details in [docs/benchmarks.md](docs/benchmarks.md)): with a ConvNet, random-network features match
features of a *trained* network at about 1/30 of the selection cost; with other architectures (ResNet-18, VGG-11,
AlexNet) a trained network's features give better coresets; synthetic-data condensation (e.g. distribution matching)
is more accurate than any selection method at small budgets.

## Documentation

[Methods](docs/methods.md) | [API](docs/api.md) | [Benchmarks](docs/benchmarks.md) | [Reproducing the numbers](docs/reproducing.md) |
[Limitations](docs/limitations.md) | [FAQ](docs/faq.md)

## Citation

If you use this software, please cite it using `CITATION.cff` (GitHub's "Cite this repository" button).

## License

MIT. See [LICENSE](LICENSE).
