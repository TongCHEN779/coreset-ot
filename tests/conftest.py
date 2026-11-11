import pytest
import torch

torch.set_num_threads(2)


@pytest.fixture(scope="session")
def blobs():
    """Small clustered "images": 4 classes, each a mixture of 3 well-separated blobs; (x, y)."""
    g = torch.Generator().manual_seed(0)
    n_per, classes = 90, 4
    xs, ys = [], []
    for c in range(classes):
        centers = torch.randn(3, 3 * 8 * 8, generator=g) * 2.0
        comp = torch.randint(0, 3, (n_per,), generator=g)
        x = centers[comp] + 0.5 * torch.randn(n_per, 3 * 8 * 8, generator=g)
        xs.append(x.view(n_per, 3, 8, 8) + 0.3 * c)
        ys.append(torch.full((n_per,), c))
    return torch.cat(xs), torch.cat(ys)


@pytest.fixture()
def points():
    g = torch.Generator().manual_seed(1)
    X = torch.randn(60, 5, generator=g)
    X[:30] += 1.5
    return X, torch.cdist(X, X)
