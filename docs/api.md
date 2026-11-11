# API reference

## `coreset_ot.select_coreset`
```python
select_coreset(data=None, labels=None, ipc=10, *, method="hybrid", features="random_convnet", n_nets=1, seed=0,
               device=None, hybrid_fraction=0.5, max_exact=12000, quality=True, verbose=False) -> CoresetResult
```
| argument | meaning |
|---|---|
| `data` | images `(N, C, H, W)` (tensor or array), normalised. Not needed for precomputed features or `method="random", quality=False` |
| `labels` | integer class labels `(N,)` |
| `ipc` | examples per class; a class with fewer examples is taken whole |
| `method` | `hybrid`, `kmedian`, `herding`, `kcenter`, `kmedoids`, `facility_location`, `random` |
| `features` | `"random_convnet"`, `"random_resnet18"`, `"random_vgg11"`, `"random_alexnet"`, `"random_mixed"`, `"pixels"`; a callable `x -> (n, d)` array or list of arrays; or precomputed `(N, d)` arrays |
| `n_nets` | number of random networks (feature blocks) for `random_*` names |
| `seed` | seed for the random networks and for the randomised solvers |
| `hybrid_fraction` | share of the budget given to k-median in `hybrid` |
| `max_exact` | classes larger than this are subsampled to a random pool of this size (distance matrices are `n x n`) |
| `quality` | compute per-class `w1`, `eps`, `mmd2` and Voronoi weights |

Networks with batch normalisation (ResNet-18, VGG-11) use batch statistics when extracting features, so their
features depend mildly on the batch composition (batch size 1000).

## `CoresetResult`
`indices`, `labels`, `weights`, `method`, `ipc`, `params`, `per_class`; methods `subset(data, labels=None)`,
`loss_weights()`, `summary()`, `save(path)`, `CoresetResult.load(path)`.

## Features
`RandomNetFeatures(arch, n_nets, seed, device, batch_size)`, `MixedArchFeatures(archs, seed, device)`,
`PixelFeatures()`. An extractor is any callable returning a list of `(n, d_k)` tensors.

## Metrics
`quality_report(blocks, idx)`, `mean_assignment_distance`, `covering_radius`, `mmd2`, `voronoi_weights` (all take
a list of feature blocks and selected row indices).

## Solvers (`coreset_ot.solvers`)
`kcenter_greedy`, `facility_location_greedy`, `pam_swap`, `kmedian`, `kmedian_cost`, `kmedoids_alternating`,
`herding`, `hybrid`: functions on a distance matrix or a feature matrix, returning lists of row indices.

## Evaluation (`coreset_ot.eval`)
`train_and_evaluate(train_x, train_y, test_x, test_y, *, arch, seed, epochs, aug, weights, device)` trains a network
from scratch and returns test accuracy; `evaluate_coreset(result, ...)` does so for several seeds;
`default_epochs(n)` is the epoch rule `min(300, ceil(2400 / steps_per_epoch))`; `dsa`, `strategy_for` expose the
augmentation. `aug` is `None`, `"dsa"` or `"simple"`.

## Datasets (`coreset_ot.datasets`, needs the `datasets` extra)
`load_dataset(name, root, download)` for `mnist`, `fmnist`, `cifar10`, `cifar100`, `tiny-imagenet`.
