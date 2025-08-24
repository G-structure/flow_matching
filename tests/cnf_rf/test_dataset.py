import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from cnf_rf.data import IQDataset


def _create_npy(tmp_path):
    arr = (np.random.randn(2, 8) + 1j * np.random.randn(2, 8)).astype(np.complex64)
    path = tmp_path / "sample.npy"
    np.save(path, arr)
    return arr, path


def test_dataset_len_and_shapes(tmp_path):
    arr, _ = _create_npy(tmp_path)
    ds = IQDataset(str(tmp_path), max_jitter=0, cond_dim=3)
    assert len(ds) == 1
    x, cond = ds[0]
    assert x.shape == (2, 8, 2)
    expected = torch.view_as_real(torch.from_numpy(arr)).float()
    assert torch.allclose(x, expected)
    assert cond.shape == (3,)


def test_dataset_applies_jitter(tmp_path, monkeypatch):
    arr, _ = _create_npy(tmp_path)
    ds = IQDataset(str(tmp_path), max_jitter=5, cond_dim=0)
    monkeypatch.setattr(np.random, "randint", lambda low, high: 2)
    x, _ = ds[0]
    expected = torch.roll(torch.view_as_real(torch.from_numpy(arr)), shifts=2, dims=1).float()
    assert torch.allclose(x, expected)
