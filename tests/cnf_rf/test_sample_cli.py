import os
import sys
import subprocess
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from cnf_rf.model import CNF
from cnf_rf.data import IQBurstDataset


def make_dataset(tmp_path):
    class_dirs = [tmp_path / "c0", tmp_path / "c1"]
    for i, d in enumerate(class_dirs):
        d.mkdir(parents=True)
        arr = (np.random.randn(2, 8) + 1j * np.random.randn(2, 8)).astype(np.complex64)
        np.save(d / f"sample{i}.npy", arr)
    return tmp_path


def create_checkpoint(path):
    model = CNF(channels=2, hidden=4, blocks=1, cond_dim=2)
    ckpt = {
        "state_dict": model.state_dict(),
        "cond_dim": 2,
        "channels": 2,
        "hidden": 4,
        "blocks": 1,
    }
    torch.save(ckpt, path)
    return path


def test_cli_sample(tmp_path):
    ckpt = create_checkpoint(tmp_path / "ckpt.pt")
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)
    cmd = [
        sys.executable,
        "-m",
        "cnf_rf.sample_cnf",
        "--checkpoint",
        str(ckpt),
        "--mode",
        "sample",
        "--num",
        "2",
        "--length",
        "8",
    ]
    subprocess.run(cmd, check=True, cwd=tmp_path, env=env)
    arr = np.load(tmp_path / "samples.npy")
    assert arr.shape == (2, 2, 8, 2)


def test_cli_log_likelihood(tmp_path):
    data_dir = make_dataset(tmp_path / "data")
    ckpt = create_checkpoint(tmp_path / "ckpt.pt")
    file_path = data_dir / "c0" / "sample0.npy"
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)
    cmd = [
        sys.executable,
        "-m",
        "cnf_rf.sample_cnf",
        "--checkpoint",
        str(ckpt),
        "--mode",
        "ll",
        "--file",
        str(file_path),
    ]
    proc = subprocess.run(cmd, check=True, cwd=tmp_path, env=env, capture_output=True, text=True)
    assert "log-likelihood" in proc.stdout
