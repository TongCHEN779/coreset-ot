"""Small image classifiers used as (a) random feature extractors and (b) evaluation networks.

All models expose ``embed(x)`` returning the flattened pre-classifier feature map, and ``forward(x)`` logits.
Architectures follow the dataset-condensation benchmarks: ConvNet-3 with instance normalisation, a CIFAR-style
ResNet-18, VGG-11 with batch normalisation, and a small-image AlexNet.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

ARCHS = ("convnet", "resnet18", "vgg11", "alexnet")
_BATCHNORM_ARCHS = ("resnet18", "vgg11")


class ConvNet(nn.Module):
    """``[conv3x3 - instance norm - ReLU - avgpool2] x depth`` followed by a linear classifier."""

    def __init__(self, in_channels: int, num_classes: int, image_size: int, width: int = 128, depth: int = 3):
        super().__init__()
        layers, c, size = [], in_channels, image_size
        for _ in range(depth):
            layers += [nn.Conv2d(c, width, 3, padding=1), nn.GroupNorm(width, width, affine=True),
                       nn.ReLU(inplace=True), nn.AvgPool2d(2, 2)]
            c, size = width, size // 2
        self.features = nn.Sequential(*layers)
        self.fc = nn.Linear(width * size * size, num_classes)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.embed(x))


class _BasicBlock(nn.Module):
    def __init__(self, i: int, o: int, stride: int):
        super().__init__()
        self.c1 = nn.Conv2d(i, o, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(o)
        self.c2 = nn.Conv2d(o, o, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(o)
        self.sh = nn.Sequential() if stride == 1 and i == o else nn.Sequential(
            nn.Conv2d(i, o, 1, stride, bias=False), nn.BatchNorm2d(o))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(y + self.sh(x))


class ResNet18(nn.Module):
    """CIFAR-style ResNet-18 (3x3 stem, no max-pool) with batch normalisation."""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_channels, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU())
        cfg, layers, i = [(64, 1), (64, 1), (128, 2), (128, 1), (256, 2), (256, 1), (512, 2), (512, 1)], [], 64
        for o, s in cfg:
            layers.append(_BasicBlock(i, o, s))
            i = o
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(512, num_classes)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(self.stem(x)).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(F.adaptive_avg_pool2d(self.layers(self.stem(x)), 1).flatten(1))


class VGG11(nn.Module):
    """VGG-11 with batch normalisation and global average pooling (max-pools are skipped once the map is 1x1)."""

    def __init__(self, in_channels: int, num_classes: int, image_size: int = 32):
        super().__init__()
        cfg, layers, i, size = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"], [], in_channels, image_size
        for v in cfg:
            if v == "M":
                if size >= 2:                      # skip pooling once the map is 1x1 (images smaller than 32 pixels)
                    layers.append(nn.MaxPool2d(2, 2))
                    size //= 2
            else:
                layers += [nn.Conv2d(i, v, 3, padding=1), nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                i = v
        self.features = nn.Sequential(*layers)
        self.fc = nn.Linear(512, num_classes)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(F.adaptive_avg_pool2d(self.features(x), 1).flatten(1))


class AlexNetS(nn.Module):
    """AlexNet-style network for 28/32 pixel inputs (no batch normalisation)."""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 128, 5, 1, 2), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 192, 5, 1, 2), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(192, 256, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(256, 192, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, 3, 1, 1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2))
        self.fc = nn.Linear(192, num_classes)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(F.adaptive_avg_pool2d(self.features(x), 1).flatten(1))


def uses_batchnorm(arch: str) -> bool:
    return arch in _BATCHNORM_ARCHS


def make_model(arch: str, in_channels: int, num_classes: int, image_size: int, depth=None) -> nn.Module:
    """Build one of :data:`ARCHS`. ``depth`` (ConvNet only) defaults to 4 for images of 64 pixels or more."""
    if arch == "convnet":
        return ConvNet(in_channels, num_classes, image_size, depth=depth or (4 if image_size >= 64 else 3))
    if arch == "resnet18":
        return ResNet18(in_channels, num_classes)
    if arch == "vgg11":
        return VGG11(in_channels, num_classes, image_size)
    if arch == "alexnet":
        return AlexNetS(in_channels, num_classes)
    raise ValueError(f"unknown architecture {arch!r}; choose from {ARCHS}")
