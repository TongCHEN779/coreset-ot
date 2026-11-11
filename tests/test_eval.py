import math

import pytest
import torch
import torch.nn.functional as F

from coreset_ot import select_coreset
from coreset_ot.eval import dsa, evaluate_coreset, strategy_for, train_and_evaluate
from coreset_ot.eval.augment import FAMILIES, affine_warp, simple_augment
from coreset_ot.eval.models import ARCHS, make_model
from coreset_ot.eval.train import default_epochs


def test_affine_warp_matches_grid_sample_forward_and_backward():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(6, 3, 12, 12, generator=g)
    a = (torch.rand(6, generator=g) - 0.5) * 0.6
    s = torch.rand(6, generator=g) * 0.4 + 0.8
    th = torch.zeros(6, 2, 3)
    th[:, 0, 0], th[:, 0, 1], th[:, 1, 0], th[:, 1, 1] = s * a.cos(), -a.sin(), a.sin(), s * a.cos()
    th[:, :, 2] = (torch.rand(6, 2, generator=g) - 0.5) * 0.2
    ref = F.grid_sample(x, F.affine_grid(th, x.shape, align_corners=True), padding_mode="zeros", align_corners=True)
    assert torch.allclose(affine_warp(x, th), ref, atol=1e-5)
    x1, x2 = x.clone().requires_grad_(True), x.clone().requires_grad_(True)
    affine_warp(x1, th).pow(2).sum().backward()
    F.grid_sample(x2, F.affine_grid(th, x.shape, align_corners=True), padding_mode="zeros", align_corners=True).pow(2).sum().backward()
    assert torch.allclose(x1.grad, x2.grad, atol=1e-4)


@pytest.mark.parametrize("name", sorted(FAMILIES))
@pytest.mark.parametrize("shared", [False, True])
def test_every_augmentation_family_keeps_shape_and_is_differentiable(name, shared):
    x = torch.randn(5, 3, 16, 16, requires_grad=True)
    y = FAMILIES[name](x, shared)
    assert y.shape == x.shape
    y.sum().backward()
    assert x.grad is not None


def test_shared_parameters_apply_the_same_transform_to_every_sample():
    x = torch.randn(1, 3, 16, 16).repeat(4, 1, 1, 1)
    y = FAMILIES["rotate"](x, True)
    assert all(torch.allclose(y[0], y[i], atol=1e-5) for i in range(4))


def test_strategy_and_dsa_dispatch():
    assert "cutout" in strategy_for(3) and "cutout" not in strategy_for(1)
    assert dsa(torch.randn(2, 1, 16, 16), strategy_for(1)).shape == (2, 1, 16, 16)
    assert simple_augment(torch.randn(3, 3, 16, 16), flip=True).shape == (3, 3, 16, 16)


@pytest.mark.parametrize("arch", ARCHS)
def test_models_forward_and_embed(arch):
    m = make_model(arch, 3, 5, 16)
    x = torch.randn(4, 3, 16, 16)
    assert m(x).shape == (4, 5) and m.embed(x).ndim == 2
    with pytest.raises(ValueError):
        make_model("bogus", 3, 5, 16)


def test_default_epochs_rule():
    assert default_epochs(100) == 300 and default_epochs(500) == 300
    assert default_epochs(5000) == math.ceil(2400 / 20) == 120
    assert default_epochs(50000) == 13


@pytest.fixture(scope="module")
def tiny_task():
    g = torch.Generator().manual_seed(0)
    y = torch.arange(400) % 4
    x = torch.randn(400, 1, 12, 12, generator=g) * 0.3
    x[:, :, :6, :6] += (y == 0).float().view(-1, 1, 1, 1)           # class-dependent patterns -> learnable
    x[:, :, 6:, :6] += (y == 1).float().view(-1, 1, 1, 1)
    x[:, :, :6, 6:] += (y == 2).float().view(-1, 1, 1, 1)
    x[:, :, 6:, 6:] += (y == 3).float().view(-1, 1, 1, 1)
    return x[:300], y[:300], x[300:], y[300:]


@pytest.mark.parametrize("aug", [None, "dsa", "simple"])
def test_training_learns_a_trivial_task(tiny_task, aug):
    xtr, ytr, xte, yte = tiny_task
    acc = train_and_evaluate(xtr, ytr, xte, yte, arch="convnet", epochs=6, aug=aug, device="cpu", seed=0)
    assert 0.0 <= acc <= 1.0 and acc > 0.5


def test_weights_and_bad_aug(tiny_task):
    xtr, ytr, xte, yte = tiny_task
    acc = train_and_evaluate(xtr, ytr, xte, yte, epochs=2, weights=torch.rand(300) + 0.1, device="cpu")
    assert 0.0 <= acc <= 1.0
    with pytest.raises(ValueError):
        train_and_evaluate(xtr, ytr, xte, yte, epochs=1, aug="bogus", device="cpu")


def test_training_is_reproducible(tiny_task):
    xtr, ytr, xte, yte = tiny_task
    a = train_and_evaluate(xtr, ytr, xte, yte, epochs=2, seed=3, device="cpu")
    b = train_and_evaluate(xtr, ytr, xte, yte, epochs=2, seed=3, device="cpu")
    assert a == b


@pytest.mark.slow
def test_evaluate_coreset_end_to_end(tiny_task):
    xtr, ytr, xte, yte = tiny_task
    r = select_coreset(xtr, ytr, ipc=10, device="cpu")
    out = evaluate_coreset(r, xtr, ytr, xte, yte, seeds=(0, 1), epochs=8, device="cpu", use_weights=True)
    assert len(out["accs"]) == 2 and out["mean"] > 0.5
