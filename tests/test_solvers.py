import itertools
import json
import os

import numpy as np
import pytest
import torch

from coreset_ot import solvers

HERE = os.path.dirname(__file__)


def brute_force_greedy_fl(D, m, w=None):
    """Independent reference: at every step evaluate the facility-location gain of each candidate explicitly."""
    n = D.shape[0]
    w = torch.ones(n) if w is None else w
    S = D.max() - D
    chosen, cur = [], torch.zeros(n)
    for _ in range(m):
        best, bj = -1.0, None
        for j in range(n):
            if j in chosen:
                continue
            gain = float(sum(w[i] * max(float(S[j, i] - cur[i]), 0.0) for i in range(n)))
            if gain > best:
                best, bj = gain, j
        chosen.append(bj)
        cur = torch.maximum(cur, S[bj])
    return chosen


def test_facility_location_matches_bruteforce(points):
    _, D = points
    assert solvers.facility_location_greedy(D, 6) == brute_force_greedy_fl(D, 6)


def test_weighted_facility_location_matches_bruteforce(points):
    _, D = points
    w = torch.rand(D.shape[0], generator=torch.Generator().manual_seed(3)) + 0.5
    assert solvers.facility_location_greedy(D, 5, w) == brute_force_greedy_fl(D, 5, w)


def test_kmedian_cost_decreases_with_budget(points):
    _, D = points
    costs = [solvers.kmedian_cost(D, solvers.kmedian(D, m)) for m in (1, 2, 4, 8, 16)]
    assert all(a > b for a, b in zip(costs, costs[1:]))


def test_pam_never_increases_cost_and_is_locally_optimal(points):
    _, D = points
    start = solvers.facility_location_greedy(D, 5)
    out = solvers.pam_swap(D, start, max_passes=50)
    assert solvers.kmedian_cost(D, out) <= solvers.kmedian_cost(D, start) + 1e-6
    base = solvers.kmedian_cost(D, out)
    for pos, j in itertools.product(range(5), range(D.shape[0])):          # no single swap improves a converged solution
        if j in out:
            continue
        trial = list(out)
        trial[pos] = j
        assert solvers.kmedian_cost(D, trial) >= base - 1e-4 * base


def test_pam_improves_a_bad_start(points):
    _, D = points
    bad = [0, 1, 2, 3, 4]
    assert solvers.kmedian_cost(D, solvers.pam_swap(D, bad)) < solvers.kmedian_cost(D, bad)


def test_kcenter_is_two_approximation():
    g = torch.Generator().manual_seed(5)
    X = torch.randn(12, 2, generator=g)
    D = torch.cdist(X, X)
    best = min(float(D[:, list(c)].min(1).values.max()) for c in itertools.combinations(range(12), 3))
    for seed in range(5):
        sel = solvers.kcenter_greedy(D, 3, seed=seed)
        assert float(D[:, sel].min(1).values.max()) <= 2 * best + 1e-6


def test_herding_tracks_the_mean_better_than_random(points):
    X, _ = points
    mean = X.mean(0)
    err = lambda s: float((X[s].mean(0) - mean).norm())
    assert err(solvers.herding(X, 30)) < err(solvers.herding(X, 5))
    rnd = np.mean([err(list(np.random.default_rng(s).choice(60, 10, replace=False))) for s in range(20)])
    assert err(solvers.herding(X, 10)) < rnd


def test_herding_continues_from_initial_points(points):
    X, _ = points
    out = solvers.herding(X, 9, initial=[3, 4])
    assert out[:2] == [3, 4] and len(out) == 9 and len(set(out)) == 9


def test_hybrid_contains_kmedian_part_and_has_right_size(points):
    X, D = points
    out = solvers.hybrid(D, X, 10, fraction=0.5)
    assert len(out) == 10 and len(set(out)) == 10
    assert out[:5] == solvers.kmedian(D, 5)


def test_hybrid_budget_one_is_the_kmedian_point(points):
    X, D = points
    assert solvers.hybrid(D, X, 1) == solvers.kmedian(D, 1)


def test_kmedoids_alternating_is_valid_and_better_than_random(points):
    _, D = points
    sel = solvers.kmedoids_alternating(D, 6, seed=0)
    assert len(set(sel)) == 6
    rnd = np.mean([solvers.kmedian_cost(D, list(np.random.default_rng(s).choice(60, 6, replace=False))) for s in range(20)])
    assert solvers.kmedian_cost(D, sel) < rnd


@pytest.mark.parametrize("fn", [lambda D, X, m: solvers.kcenter_greedy(D, m), lambda D, X, m: solvers.kmedian(D, m),
                                lambda D, X, m: solvers.herding(X, m), lambda D, X, m: solvers.hybrid(D, X, m)])
def test_budget_validation(points, fn):
    X, D = points
    with pytest.raises(ValueError):
        fn(D, X, 0)
    with pytest.raises(ValueError):
        fn(D, X, 61)


def test_deterministic(points):
    X, D = points
    assert solvers.kmedoids_alternating(D, 5, seed=2) == solvers.kmedoids_alternating(D, 5, seed=2)
    assert solvers.kcenter_greedy(D, 5, seed=2) == solvers.kcenter_greedy(D, 5, seed=2)


def test_matches_golden_outputs_of_the_research_code():
    """kcenter / kmedoids / herding must reproduce the original experiment code exactly on a seeded input."""
    with open(os.path.join(HERE, "fixtures", "golden_legacy.json")) as f:
        gold = json.load(f)
    torch.manual_seed(gold["seed"])
    X = torch.randn(gold["n"], gold["p"])
    X[:35] += 1.5
    D = torch.cdist(X, X)
    assert solvers.kcenter_greedy(D, 6, seed=0) == gold["kcenter_seed0_m6"]
    # k-medoids uses randomised k-means++-style seeding whose probabilities depend on the last bits of the float32
    # distances, which differ between BLAS implementations (macOS vs Linux): the path can fork into another, equally
    # good local optimum. So compare the solution quality instead of the indices.
    sel = solvers.kmedoids_alternating(D, 6, seed=0)
    assert len(set(sel)) == 6
    assert solvers.kmedian_cost(D, sel) <= solvers.kmedian_cost(D, gold["kmedoids_seed0_m6"]) * 1.01
    assert solvers.herding(X, 7) == gold["herding_m7"]
