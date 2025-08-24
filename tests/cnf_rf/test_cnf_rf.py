import numpy as np
import torch
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from cnf_rf.data import IQBurstDataset
from cnf_rf.model import CNF


def make_dataset(tmp_path):
    class_dirs = [tmp_path / "c0", tmp_path / "c1"]
    for i, d in enumerate(class_dirs):
        d.mkdir()
        arr = (np.random.randn(2, 16) + 1j * np.random.randn(2, 16)).astype(np.complex64)
        np.save(d / f"sample{i}.npy", arr)
    return str(tmp_path)


def test_dataset_loading(tmp_path):
    data_dir = make_dataset(tmp_path)
    ds = IQBurstDataset(data_dir)
    assert ds.cond_dim == 2
    x, cond = ds[0]
    assert x.shape == (2, 16, 2)
    assert cond.shape == (2,)
    vec = ds.cond_for_path(ds.files[0][0])
    assert torch.allclose(vec, cond)


def test_cnf_forward_and_sample(tmp_path):
    data_dir = make_dataset(tmp_path)
    ds = IQBurstDataset(data_dir)
    x, cond = ds[0]
    x = x.unsqueeze(0).requires_grad_(True)
    cond = cond.unsqueeze(0)
    model = CNF(channels=2, hidden=4, blocks=1, cond_dim=2)
    drift = model.vf(x, torch.tensor(0.0), cond)
    div = torch.autograd.grad(drift, x, torch.ones_like(drift), create_graph=True)[0]
    assert drift.shape == x.shape
    assert div.shape == x.shape
    samples = model.sample(3, length=x.shape[2], cond=cond.repeat(3,1), method="euler")
    assert samples.shape == (3, 2, x.shape[2], 2)


def test_dataset_time_jitter(tmp_path):
    data_dir = make_dataset(tmp_path)
    ds = IQBurstDataset(data_dir, jitter=3)
    path, _ = ds.files[0]
    arr = np.load(path)
    orig = np.stack([arr.real, arr.imag], axis=-1).astype(np.float32)
    np.random.seed(0)
    x, _ = ds[0]
    np.random.seed(0)
    shift = np.random.randint(-3, 4)
    expected = torch.from_numpy(np.roll(orig, shift, axis=1))
    assert torch.allclose(x, expected)


def test_odefunc_no_grad_zero_divergence(tmp_path):
    data_dir = make_dataset(tmp_path)
    ds = IQBurstDataset(data_dir)
    x, cond = ds[0]
    x = x.unsqueeze(0)
    cond = cond.unsqueeze(0)
    model = CNF(channels=2, hidden=4, blocks=1, cond_dim=2)
    model.odefunc.set_condition(cond)
    t = torch.tensor(0.0)
    with torch.no_grad():
        _, neg_div = model.odefunc(t, (x, torch.zeros(1)))
    assert torch.allclose(neg_div, torch.zeros(1))


def test_loss_backward_has_gradients(tmp_path):
    data_dir = make_dataset(tmp_path)
    ds = IQBurstDataset(data_dir)
    x, cond = ds[0]
    x = x.unsqueeze(0).requires_grad_(True)
    cond = cond.unsqueeze(0)
    model = CNF(channels=2, hidden=4, blocks=1, cond_dim=2)
    loss = model.loss(x, cond)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() for g in grads)
