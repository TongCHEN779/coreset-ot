"""Use your own embeddings (e.g. a pretrained model) instead of random networks."""
import numpy as np

from coreset_ot import select_coreset

rng = np.random.default_rng(0)
labels = rng.integers(0, 5, size=2000)
embeddings = rng.normal(size=(2000, 64)) + labels[:, None]        # stand-in for real embeddings

result = select_coreset(None, labels, ipc=20, features=embeddings, method="hybrid")
print(len(result), "selected;", {k: round(v, 3) for k, v in result.summary().items()})
