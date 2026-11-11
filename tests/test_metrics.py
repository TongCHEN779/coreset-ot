import numpy as np
import pytest
import torch

from coreset_ot import covering_radius, mean_assignment_distance, mmd2, quality_report, voronoi_weights
from coreset_ot.metrics import distances_to, feature_scales


@pytest.fixture()
def blocks():
    g = torch.Generator().manual_seed(0)
    return [torch.randn(50, 6, generator=g), torch.randn(50, 9, generator=g)]


def test_distances_match_bruteforce(blocks):
    sc = feature_scales(blocks)
    sel = [3, 10, 22]
    D = distances_to(blocks, sel, sc)
    for i in (0, 7, 31):
        for k, j in enumerate(sel):
            ref = np.sqrt(np.mean([(float((f[i] - f[j]).norm()) / s) ** 2 for f, s in zip(blocks, sc)]))
            assert float(D[i, k]) == pytest.approx(ref, rel=1e-5)


def test_selected_points_have_zero_distance_to_themselves(blocks):
    D = distances_to(blocks, [4, 8], feature_scales(blocks))
    assert D[4, 0] == pytest.approx(0, abs=1e-5) and D[8, 1] == pytest.approx(0, abs=1e-5)


def test_selecting_everything_gives_zero_cost_and_zero_mmd(blocks):
    allidx = list(range(50))
    assert mean_assignment_distance(blocks, allidx) == pytest.approx(0, abs=1e-5)
    assert covering_radius(blocks, allidx) == pytest.approx(0, abs=1e-5)
    assert mmd2(blocks, allidx) == pytest.approx(0, abs=1e-8)


def test_voronoi_weights_sum_to_one_and_counts(blocks):
    sel = [1, 9, 17, 30]
    w = voronoi_weights(blocks, sel)
    assert float(w.sum()) == pytest.approx(1.0) and (w >= 0).all()
    D = distances_to(blocks, sel, feature_scales(blocks))
    assert np.allclose(w.numpy() * 50, np.bincount(D.argmin(1).numpy(), minlength=4))


def test_mean_never_exceeds_radius_and_more_points_help(blocks):
    sc = feature_scales(blocks)
    small, big = list(range(5)), list(range(20))
    assert mean_assignment_distance(blocks, small, sc) <= covering_radius(blocks, small, sc)
    assert mean_assignment_distance(blocks, big, sc) < mean_assignment_distance(blocks, small, sc)


def test_quality_report_keys(blocks):
    r = quality_report(blocks, [0, 5, 9])
    assert set(r) == {"w1", "eps", "mmd2"} and r["w1"] <= r["eps"]
