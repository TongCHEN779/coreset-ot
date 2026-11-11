import json

import numpy as np
import pytest
import torch

from coreset_ot import METHODS, CoresetResult, select_coreset

KW = dict(device="cpu")


@pytest.mark.parametrize("method", METHODS)
def test_every_method_is_class_balanced_and_valid(blobs, method):
    x, y = blobs
    r = select_coreset(x, y, ipc=6, method=method, **KW)
    assert len(r) == 24 and np.bincount(r.labels).tolist() == [6] * 4
    assert np.array_equal(y.numpy()[r.indices], r.labels)                      # labels match the global indices
    for c in range(4):
        idx = r.indices[r.labels == c]
        assert len(set(idx.tolist())) == 6                                      # no duplicates
        assert float(r.weights[r.labels == c].sum()) == pytest.approx(1.0, abs=1e-5)


def test_loss_weights_give_every_class_total_weight_ipc(blobs):
    x, y = blobs
    r = select_coreset(x, y, ipc=5, **KW)
    lw = r.loss_weights()
    assert [round(float(lw[r.labels == c].sum()), 6) for c in range(4)] == [5.0] * 4


def test_selection_is_deterministic_and_seed_dependent(blobs):
    x, y = blobs
    a = select_coreset(x, y, ipc=4, method="kcenter", seed=1, **KW)
    b = select_coreset(x, y, ipc=4, method="kcenter", seed=1, **KW)
    c = select_coreset(x, y, ipc=4, method="kcenter", seed=2, **KW)
    assert np.array_equal(a.indices, b.indices) and not np.array_equal(a.indices, c.indices)


def test_kmedian_beats_kcenter_and_random_on_mean_distance(blobs):
    x, y = blobs
    w1 = {m: select_coreset(x, y, ipc=6, method=m, features="random_convnet", **KW).summary()["w1"] for m in ("kmedian", "kcenter", "random")}
    assert w1["kmedian"] < w1["kcenter"] and w1["kmedian"] < w1["random"]


def test_herding_mmd_decreases_with_budget_and_beats_average_random(blobs):
    x, y = blobs
    h = {m: select_coreset(x, y, ipc=m, method="herding", **KW).summary()["mmd2"] for m in (4, 20)}
    rnd = np.mean([select_coreset(x, y, ipc=20, method="random", seed=s, **KW).summary()["mmd2"] for s in range(8)])
    assert h[20] < h[4] and h[20] < rnd


def test_ipc_larger_than_class_takes_whole_class():
    x = torch.randn(10, 1, 8, 8)
    y = torch.tensor([0] * 4 + [1] * 6)
    r = select_coreset(x, y, ipc=5, **KW)
    assert np.bincount(r.labels).tolist() == [4, 5]


def test_precomputed_features_and_custom_extractor(blobs):
    x, y = blobs
    feats = x.flatten(1)
    a = select_coreset(None, y, ipc=4, features=feats, method="hybrid", **KW)
    b = select_coreset(x, y, ipc=4, features=lambda z: z.flatten(1), method="hybrid", **KW)
    assert np.array_equal(a.indices, b.indices)
    c = select_coreset(None, y, ipc=4, features=[feats, feats * 2], method="kmedian", **KW)
    assert len(c) == 16


def test_error_cases(blobs):
    x, y = blobs
    with pytest.raises(ValueError):
        select_coreset(x, y, method="bogus", **KW)
    with pytest.raises(ValueError):
        select_coreset(x, None, **KW)
    with pytest.raises(ValueError):
        select_coreset(None, y, **KW)
    with pytest.raises(ValueError):
        select_coreset(None, y, features=torch.randn(5, 3), **KW)


def test_large_class_is_subsampled_with_a_warning(blobs):
    x, y = blobs
    with pytest.warns(RuntimeWarning):
        r = select_coreset(x, y, ipc=3, method="kmedian", max_exact=40, **KW)
    assert np.bincount(r.labels).tolist() == [3] * 4


def test_save_load_roundtrip(blobs, tmp_path):
    x, y = blobs
    r = select_coreset(x, y, ipc=3, **KW)
    p = str(tmp_path / "r.json")
    r.save(p)
    r2 = CoresetResult.load(p)
    assert np.array_equal(r.indices, r2.indices) and r2.method == "hybrid" and r2.ipc == 3
    with open(p) as f:
        assert json.load(f)["params"]["features"] == "random_convnet"
    assert set(r2.per_class[0]) >= {"w1", "eps", "mmd2"}


def test_subset_helper(blobs):
    x, y = blobs
    r = select_coreset(x, y, ipc=3, **KW)
    sx, sy = r.subset(x, y)
    assert sx.shape[0] == 12 and torch.equal(sy, torch.as_tensor(r.labels))


def test_numpy_inputs_work(blobs):
    x, y = blobs
    r = select_coreset(x.numpy(), y.numpy(), ipc=3, **KW)
    assert len(r) == 12


def test_random_method_without_quality_needs_no_features(blobs):
    _, y = blobs
    r = select_coreset(None, y, ipc=3, method="random", quality=False)
    assert len(r) == 12 and r.per_class == {}
