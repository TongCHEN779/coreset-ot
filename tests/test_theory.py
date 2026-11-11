"""Checks of the mathematical statements the package relies on."""
import numpy as np
import pytest
import torch
from scipy.optimize import linprog

from coreset_ot import solvers
from coreset_ot.distances import ensemble_distance


def exact_w1(D, sel, lam):
    """Exact W1 between the uniform measure on all points and sum_j lam_j delta_{sel_j} by linear programming."""
    n, m = D.shape[0], len(sel)
    C = D[:, sel].numpy().ravel()
    A, b = [], []
    for i in range(n):
        r = np.zeros(n * m)
        r[i * m:(i + 1) * m] = 1
        A.append(r)
        b.append(1.0 / n)
    for j in range(m):
        r = np.zeros(n * m)
        r[j::m] = 1
        A.append(r)
        b.append(lam[j])
    return linprog(C, A_eq=np.array(A), b_eq=b, bounds=(0, None), method="highs").fun


@pytest.fixture()
def small():
    g = torch.Generator().manual_seed(11)
    X = torch.randn(40, 3, generator=g)
    return torch.cdist(X, X)


def test_voronoi_weights_attain_mean_assignment_distance(small):
    D = small
    sel = solvers.kcenter_greedy(D, 4, seed=0)
    near = D[:, sel].min(1).values
    lam = torch.bincount(D[:, sel].argmin(1), minlength=4).double().numpy() / 40
    assert exact_w1(D, sel, lam) == pytest.approx(float(near.mean()), abs=1e-6)


def test_no_other_weights_beat_voronoi(small):
    D = small
    sel = solvers.kmedian(D, 4)
    wbar = float(D[:, sel].min(1).values.mean())
    rng = np.random.default_rng(0)
    for _ in range(5):
        lam = rng.dirichlet(np.ones(4))
        assert exact_w1(D, sel, lam) >= wbar - 1e-6
    assert exact_w1(D, sel, np.ones(4) / 4) >= wbar - 1e-6


def test_mean_distance_never_exceeds_covering_radius(small):
    D = small
    for m in (2, 5, 9):
        near = D[:, solvers.kcenter_greedy(D, m, seed=1)].min(1).values
        assert float(near.mean()) <= float(near.max()) + 1e-9


def test_facility_location_objective_equals_kmedian_up_to_constant(small):
    """sum_j max_i S_ij = n * max(D) - sum_j min_i D_ij, so maximising one minimises the other."""
    D = small
    S = D.max() - D
    for sel in ([0, 5, 9], [3, 4], [10, 20, 30, 39]):
        assert float(S[sel].max(0).values.sum()) == pytest.approx(40 * float(D.max()) - float(D[:, sel].min(1).values.sum()), rel=1e-5)


def test_ensemble_bound_jensen():
    """mean_k d_k <= sqrt(mean_k d_k^2): the expected block-wise cost is bounded by the RMS-metric cost."""
    g = torch.Generator().manual_seed(2)
    blocks = [torch.randn(50, 6, generator=g) for _ in range(5)]
    D = ensemble_distance(blocks)
    sel = solvers.kmedian(D, 5)
    assign = D[:, sel].argmin(1)
    rms_cost = float(D[torch.arange(50), torch.tensor(sel)[assign]].mean())
    per_block = []
    for f in blocks:
        Dk = torch.cdist(f, f)
        Dk = Dk / Dk.median()
        per_block.append(float(Dk[torch.arange(50), torch.tensor(sel)[assign]].mean()))
    assert np.mean(per_block) <= rms_cost + 1e-6
