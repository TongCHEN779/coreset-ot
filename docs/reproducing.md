# Reproducing the benchmark numbers

1. Install: `pip install -e ".[dev,datasets]"`.
2. Get the datasets (torchvision downloads MNIST, Fashion-MNIST, CIFAR-10/100 with `download=True`; Tiny-ImageNet
   must be unpacked to `<root>/tiny-imagenet-200`).
3. Run `examples/reproduce_cifar10.py` (CIFAR-10, hybrid selector, 8 random networks, ConvNet, 5 seeds). The original
   runs gave 38.4 +- 0.3 % at 10 images per class and 51.2 +- 0.8 % at 50 per class for this setting.

Seeds: random network `k` of seed `s` is initialised with `7919 * s + 104729 + k` and the training seed is `s`, as in
the original experiments.

The research scripts that produced the other numbers in [benchmarks](benchmarks.md) (trained-proxy selectors,
gradient-based selectors, distribution matching, the statistics) are not part of this repository.

Measured with the released package on Apple MPS (wheel installed in a clean environment, CIFAR-10, hybrid, 8 random
ConvNets, 5 seeds): 38.4 +- 0.7 % at 10 images per class (seeds: 37.5, 38.4, 38.1, 38.5, 39.5). The 50-per-class setting was
not re-run with the package.

Small differences are expected: the original numbers were produced on Apple MPS; floating-point kernels and tie-breaking
differ across hardware.
