# Limitations

* **Architecture dependence.** Random-network features gave coresets as good as trained-network features for a
  ConvNet, but for ResNet-18, VGG-11 and AlexNet features of a *trained* network gave better coresets
  (facility location on trained features was best or tied everywhere outside the ConvNet). Averaging random networks of
  several architectures (`random_mixed`) helped on ResNet-18 and VGG-11 for CIFAR-10 but not on the ConvNet, AlexNet or
  CIFAR-100. If you train something other than a ConvNet, consider supplying features of a trained model via
  `features=your_embeddings`.
* **Selection versus synthesis.** Methods that synthesise images (e.g. distribution matching) were clearly more
  accurate than any selection method at small budgets in our pipeline (CIFAR-10, 10 images per class: 48.1% versus
  40.5% with DSA augmentation). Selected real images are interpretable and need no optimisation, but are not the most
  accurate option.
* **Scale.** Class-wise `n x n` distance matrices: exact up to `max_exact=12000` points per class; larger classes are
  subsampled to a random pool (a warning is raised). Herding does not need the matrix and works on whole classes.
* **Class-balanced only.** The same budget per class; unequal budgets are not implemented.
* **Images.** Developed and tested on small images (28-64 pixels). Other modalities work through `features=` but
  are untested.
* **Hardware.** Unit tests run on CPU; the pipeline has been run on Apple MPS. CUDA should work (only standard torch
  operations are used) but has not been validated by the authors.
* **Statistics.** The reported numbers use 3-5 seeds; differences of about one point are within seed noise.
* **Baselines.** Gradient-based selectors (GraNd, CRAIG on gradient embeddings, GradMatch) in the authors' experiments
  were simplified last-layer versions and are *not* part of this package.
