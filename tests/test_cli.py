import json

import pytest
import torch

from coreset_ot.cli import build_parser, main


@pytest.fixture()
def fake_dataset(monkeypatch):
    g = torch.Generator().manual_seed(0)
    y = torch.arange(200) % 4
    x = torch.randn(200, 1, 12, 12, generator=g) * 0.3
    x[:, :, :6] += (y % 2).float().view(-1, 1, 1, 1)
    x[:, :, :, :6] += (y // 2).float().view(-1, 1, 1, 1)
    data = (x[:150], y[:150], x[150:], y[150:])
    monkeypatch.setattr("coreset_ot.datasets.load_dataset", lambda *a, **k: data)


def test_methods_and_version(capsys):
    assert main(["methods"]) == 0
    assert "hybrid" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--version"])


def test_select_quality_evaluate(fake_dataset, tmp_path, capsys):
    out = str(tmp_path / "sel.json")
    assert main(["select", "--dataset", "mnist", "--ipc", "4", "--out", out, "--device", "cpu"]) == 0
    with open(out) as f:
        assert len(json.load(f)["indices"]) == 16
    assert main(["quality", "--dataset", "mnist", "--indices", out, "--n-nets", "2", "--device", "cpu"]) == 0
    assert "class_averaged" in capsys.readouterr().out
    assert main(["evaluate", "--dataset", "mnist", "--indices", out, "--seeds", "1", "--epochs", "2", "--device", "cpu"]) == 0
    last = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert 0 <= last["mean_acc"] <= 100


def test_unknown_method_is_rejected():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["select", "--dataset", "mnist", "--out", "x", "--method", "bogus"])
