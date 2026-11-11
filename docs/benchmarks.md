# Benchmarks

Numbers come from the authors' experiments (single Apple-silicon GPU; ConvNet-3 trained from scratch for
`min(300, ceil(2400/steps_per_epoch))` epochs; mean over seeds; the seed changes both the selection and the network
initialisation). They document what the selectors did in that study; treat differences of about one point as noise.

## ConvNet, no augmentation (images per class: 1 / 10 / 50)
Random-network features, 8 networks (Tiny-ImageNet: 1 network, 3 seeds; CIFAR-100: 3 seeds; others 5).

| Dataset | Random | k-center (pixels) | Trained-net herding | Trained-net facility loc. + weights | k-median | Herding | Hybrid |
|---|---|---|---|---|---|---|---|
| MNIST | 61.3 / 91.1 / 96.6 | 61.3 / 89.2 / 96.3 | 82.1 / 94.4 / 97.2 | 81.2 / 93.3 / 96.5 | 84.8 / 94.0 / 96.4 | 75.0 / 94.4 / 97.0 | 84.8 / 94.2 / 97.1 |
| FMNIST | 50.3 / 74.5 / 81.9 | 50.3 / 56.7 / 65.0 | 65.8 / 77.7 / 84.1 | 67.9 / 79.0 / 84.8 | 68.4 / 79.3 / 85.0 | 66.8 / 79.1 / 84.9 | 68.4 / 80.3 / 85.3 |
| CIFAR-10 | 14.7 / 27.4 / 42.8 | 14.7 / 18.6 / 34.1 | 20.0 / 35.4 / 49.4 | 21.4 / 36.4 / 48.1 | 20.4 / 36.9 / 48.0 | 22.5 / 38.0 / 51.0 | 20.4 / 38.4 / 51.2 |
| CIFAR-100 | 4.0 / 14.4 / 29.3 | 4.0 / 10.2 / 24.8 | 8.6 / 19.2 / 32.8 | 8.8 / 21.1 / 34.1 | 7.7 / 21.2 / 32.7 | 8.9 / 20.4 / 34.2 | 7.7 / 21.0 / 34.4 |
| Tiny-ImageNet | 1.5 / 4.6 / 12.9 | - | - | - | 2.3 / 7.9 / 15.6 | 2.7 / 7.2 / 16.3 | 2.3 / 7.4 / 16.4 |

Herding and the hybrid were significantly better than random in all 15 dataset/budget cells (Welch test, Holm
corrected). Greedy k-center on pixels was significantly *worse* than random in 6 of 12 cells. "Trained-net" selectors
use a ConvNet trained for 20 epochs on the full data (not included in this package).

## Other architectures (CIFAR-10, 10 images per class unless stated; 5 seeds)
| Training network | Random | Trained-net facility loc. | Hybrid (ConvNet features) | Hybrid (mixed architectures) |
|---|---|---|---|---|
| ConvNet | 27.4 | 36.4 | 38.4 | 35.2 |
| ResNet-18 | 23.2 | 27.4 | 23.8 | 26.7 |
| ResNet-18, 50 per class | 32.8 | 38.7 | 34.4 | 36.7 |
| VGG-11 | 30.4 | 38.3 | 33.4 | 37.3 |
| AlexNet (noisy, std 2-7) | 22.6 | 28.4 | 24.1 | 23.1 |

Random ConvNet features are not architecture-agnostic: for other architectures, features of a trained network gave
better coresets. Mixing architectures helps on ResNet-18 and VGG-11 but not consistently (no gain on CIFAR-100).

## DSA augmentation and synthetic data (CIFAR-10; images per class 1 / 10 / 50)
| Method | 1 | 10 | 50 |
|---|---|---|---|
| Random | 14.6 | 31.1 | 48.5 |
| Herding (random-network features) | 21.9 | 40.5 | 55.0 |
| Hybrid | 20.9 | 40.2 | 54.4 |
| Distribution matching (our re-implementation, 1000 iterations) | 25.7 | 48.1 | - |

Selection with real images is below synthetic condensation at small budgets. Initialising distribution matching
from the selected images instead of random images started 9.5 points higher at 10 images per class but gave the same
accuracy after matching (48.8 / 48.0 versus 48.1).

## Cost
CIFAR-10, all 10 classes, 10 images per class, one GPU: 4.5 s (herding), 5.7 s (k-median), 6.8 s (hybrid) with one
random network, about 30 s with eight; training a 20-epoch proxy network on the full data took 178 s.

## Quality measures predict accuracy
Across about 30 selection methods per dataset/budget, `w1` and `mmd2` measured in the features of independent random
networks correlated with ConvNet accuracy (Spearman about -0.8; between -0.75 and -0.84 when any one family of methods
is removed). The covering radius (about -0.07) and pixel-space measures (about -0.6) were much weaker.
