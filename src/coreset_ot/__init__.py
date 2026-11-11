"""coreset-ot: training-free coreset selection by optimal transport.

Quick start::

    from coreset_ot import select_coreset
    result = select_coreset(images, labels, ipc=10)          # hybrid k-median + herding in random-net features
    subset_x, subset_y = result.subset(images, labels)
"""
from ._version import __version__
from .features import MixedArchFeatures, PixelFeatures, RandomNetFeatures
from .metrics import covering_radius, mean_assignment_distance, mmd2, quality_report, voronoi_weights
from .selection import METHODS, CoresetResult, select_coreset

__all__ = ["__version__", "select_coreset", "CoresetResult", "METHODS", "RandomNetFeatures", "MixedArchFeatures",
           "PixelFeatures", "quality_report", "mean_assignment_distance", "covering_radius", "mmd2", "voronoi_weights"]
