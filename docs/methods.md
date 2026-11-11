# Methods

## Problem
Given training examples with labels, pick `ipc` examples per class so that a network trained on the subset is as
good as possible. `coreset-ot` selects *real* examples, class by class, in a feature space.

## Why the mean distance, not the covering radius
Let the class have `N` points, let the selection be `m` of them, and let `r(x)` be the distance from `x` to the
nearest selected point.

* **Covering radius** `eps = max_x r(x)` is what greedy k-center minimises. It bounds the loss gap of an L-Lipschitz
  loss by `L * eps`.
* **Mean assignment distance** `w1 = mean_x r(x)` is the k-median / k-medoids objective. If every selected point
  carries the fraction of data nearest to it (its *Voronoi weight*), `w1` equals the 1-Wasserstein distance between
  the data and the weighted selection, so the loss gap is at most `L * w1`. For fixed selected points no other
  weights give a smaller distance, and `w1 <= eps`.

So the right target is `w1`. In our experiments greedy k-center is significantly *worse than random* in half of the
dataset/budget settings, because with a small budget it places points on outliers at the edge of the data.

(Facility location, `max sum_j max_i (D_max - d_ij)`, is the same problem as k-median up to a constant; the greedy
facility-location rule is the first stage of `kmedian`. The unit tests check this identity and the greedy rule
against brute force.)

## Why feature space matters more than the optimiser
No selection of `m` points can make `w1` small in high intrinsic dimension: for data whose mass in a ball of radius
`r` is at most `C r^d`, any selection satisfies `w1 >= (1/2) (2 C m)^(-1/d)`. With `d` around 15, 100 points per class
lower the bound only by a modest constant factor, so the *metric* in which the cover is computed decides how well
the points are placed. Pixels are a poor metric; features of a network are better. We found that features of a
**randomly initialised** network are as good as features of a trained one when the downstream model is a ConvNet.

## Selectors
All selectors work on the *ensemble metric*: with feature blocks `F_1..F_K` (e.g. `K` random networks),
`d(i, j) = sqrt(mean_k (|F_k[i] - F_k[j]| / s_k)^2)` where `s_k` is the median pairwise distance of block `k`.
For `K` random networks drawn from a distribution, minimising `w1` under this metric bounds the *expected* loss gap
over the random networks (Jensen). In practice one network (`K=1`) performs as well as eight.

* `kmedian`: greedy facility location, then PAM swaps (replace one selected point at a time by the best data point
  while the cost decreases). The greedy stage and the swap stage reach similar objective values; the swaps mainly
  fix the greedy solution.
* `herding`: kernel herding with the linear kernel on the concatenated, normalised features; keeps the mean of the
  selected features close to the class mean.
* `hybrid`: `kmedian` on `round(0.5 * m)` points, then herding continues from them. Coverage first, then mean
  matching.
* `kcenter`, `kmedoids`, `facility_location`, `random`: baselines.

## Quality diagnostics
`quality_report` returns `w1`, `eps` and `mmd2` in a feature space. Measured in the features of *independent*
random networks (different seeds than those used for selection), `w1` and `mmd2` predicted ConvNet accuracy across
about 30 selection methods (Spearman about -0.8, robust to removing any family of methods), whereas the covering
radius and pixel-space versions did not. They separate good selections from poor ones; they do not rank two good
selections reliably.

## Weights
`result.weights` are Voronoi weights and `result.loss_weights()` rescales them so each class has total weight `ipc`.
In our experiments using them as loss weights changed accuracy by less than one point for most settings, so they
are optional.
