import pytest
import torch

from coreset_ot import MixedArchFeatures, PixelFeatures, RandomNetFeatures
from coreset_ot.features import as_blocks, build_extractor


@pytest.fixture()
def imgs():
    return torch.randn(24, 3, 16, 16, generator=torch.Generator().manual_seed(0))


def test_random_convnet_shapes_and_blocks(imgs):
    out = RandomNetFeatures("convnet", n_nets=3, seed=0, device="cpu")(imgs)
    assert len(out) == 3 and all(b.shape == (24, 128 * 2 * 2) for b in out)


@pytest.mark.parametrize("arch", ["resnet18", "vgg11", "alexnet"])
def test_other_architectures_run(imgs, arch):
    out = RandomNetFeatures(arch, seed=0, device="cpu")(imgs)
    assert len(out) == 1 and out[0].shape[0] == 24 and out[0].ndim == 2 and torch.isfinite(out[0]).all()


def test_same_seed_same_features_different_seed_differs(imgs):
    a = RandomNetFeatures("convnet", seed=3, device="cpu")(imgs)[0]
    b = RandomNetFeatures("convnet", seed=3, device="cpu")(imgs)[0]
    c = RandomNetFeatures("convnet", seed=4, device="cpu")(imgs)[0]
    assert torch.equal(a, b) and not torch.allclose(a, c)


def test_networks_in_one_extractor_differ(imgs):
    a, b = RandomNetFeatures("convnet", n_nets=2, seed=0, device="cpu")(imgs)
    assert not torch.allclose(a, b)


def test_global_rng_state_is_untouched(imgs):
    torch.manual_seed(5)
    before = torch.rand(3)
    torch.manual_seed(5)
    RandomNetFeatures("convnet", n_nets=2, seed=9, device="cpu")(imgs)
    assert torch.equal(torch.rand(3), before)


def test_features_are_per_sample_for_eval_mode_networks(imgs):
    ext = RandomNetFeatures("convnet", seed=0, device="cpu", batch_size=5)
    full = RandomNetFeatures("convnet", seed=0, device="cpu", batch_size=1000)(imgs)[0]
    assert torch.allclose(ext(imgs)[0], full, atol=1e-5)


def test_mixed_architectures_give_one_block_each(imgs):
    out = MixedArchFeatures(seed=0, device="cpu")(imgs)
    assert len(out) == 4 and all(b.shape[0] == 24 for b in out)


def test_pixel_features(imgs):
    assert PixelFeatures("cpu")(imgs)[0].shape == (24, 3 * 16 * 16)


def test_input_validation(imgs):
    with pytest.raises(ValueError):
        RandomNetFeatures("nope")
    with pytest.raises(ValueError):
        RandomNetFeatures("convnet", device="cpu")(imgs[0])
    ext = RandomNetFeatures("convnet", device="cpu")
    ext(imgs)
    with pytest.raises(ValueError):
        ext(torch.randn(4, 1, 16, 16))


def test_as_blocks_and_build_extractor():
    assert len(as_blocks([torch.randn(5, 2), torch.randn(5, 3)], "cpu")) == 2
    with pytest.raises(ValueError):
        as_blocks([torch.randn(5, 2), torch.randn(4, 2)], "cpu")
    assert callable(build_extractor("random_convnet", 1, 0, "cpu"))
    with pytest.raises(ValueError):
        build_extractor("bogus")
