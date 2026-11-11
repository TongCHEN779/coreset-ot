# Changelog

## 0.1.0 (unreleased)
- First release: `select_coreset` with `hybrid`, `kmedian`, `herding`, `kcenter`, `kmedoids`, `facility_location`
  and `random` selectors; random-network, mixed-architecture and pixel features; custom features.
- Quality diagnostics (`quality_report`: transport cost, covering radius, MMD, Voronoi weights).
- Evaluation harness: ConvNet / ResNet-18 / VGG-11 / AlexNet, DSA and simple augmentation, the
  train-from-scratch protocol of the condensation literature.
- Dataset loaders (MNIST, Fashion-MNIST, CIFAR-10/100, Tiny-ImageNet) and a command line interface.
