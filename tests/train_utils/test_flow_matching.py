import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from flow_matching.train_utils import divergence_from_func, FlowMatchingTrainer
from cnf_rf.model import build_flow
from cnf_rf.data import IQDataset


def test_divergence_linear(monkeypatch):
    A = torch.randn(3, 3)
    def linear(x):
        return x @ A.t()
    x = torch.randn(5, 3, requires_grad=True)
    monkeypatch.setattr(torch, "randn_like", lambda t: torch.ones_like(t))
    div = divergence_from_func(lambda inp: linear(inp), x)
    expected = torch.trace(A).expand(5)
    assert torch.allclose(div, expected)


def test_trainer_runs_one_epoch(tmp_path):
    arr = (np.random.randn(1, 4) + 1j * np.random.randn(1, 4)).astype(np.complex64)
    np.save(tmp_path / "sample.npy", arr)
    ds = IQDataset(str(tmp_path))
    dl = DataLoader(ds, batch_size=1)
    model = build_flow(channels=1, cond_dim=0, num_layers=1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = FlowMatchingTrainer(model, opt, torch.device("cpu"))
    trainer.train(dl, epochs=1)
