"""Data loading utilities for CNF RF-IQ bursts."""
from __future__ import annotations

import os
import glob
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class IQBurstDataset(Dataset):
    """Dataset of complex IQ bursts stored as ``.npy`` files.

    The directory should contain subfolders for each condition. Each ``.npy``
    file is expected to have shape ``[C, T]`` with ``complex64`` dtype.
    """

    def __init__(self, data_dir: str, jitter: int = 0):
        self.data_dir = data_dir
        self.jitter = jitter
        self.files: List[Tuple[str, int]] = []
        subdirs = [d for d in sorted(os.listdir(data_dir)) if os.path.isdir(os.path.join(data_dir, d))]
        if not subdirs:
            subdirs = ["."]
        self.cond_dim = len(subdirs)
        for idx, sub in enumerate(subdirs):
            for f in glob.glob(os.path.join(data_dir, sub, "*.npy")):
                self.files.append((f, idx))

    def cond_for_path(self, file_path: str) -> torch.Tensor:
        """Return one-hot condition vector for a given file path."""
        abs_path = os.path.abspath(file_path)
        for p, label in self.files:
            if os.path.abspath(p) == abs_path:
                vec = torch.zeros(self.cond_dim, dtype=torch.float32)
                vec[label] = 1.0
                return vec
        raise ValueError(f"{file_path} not found in dataset")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path, label = self.files[idx]
        arr = np.load(path)
        if np.iscomplexobj(arr):
            arr = np.stack([arr.real, arr.imag], axis=-1)
        else:
            raise ValueError("Expected complex64 array")
        if self.jitter > 0:
            shift = np.random.randint(-self.jitter, self.jitter + 1)
            arr = np.roll(arr, shift, axis=1)
        x = torch.from_numpy(arr.astype(np.float32))
        cond = torch.zeros(self.cond_dim, dtype=torch.float32)
        cond[label] = 1.0
        return x, cond


def make_dataloader(data_dir: str, batch_size: int, jitter: int = 0, num_workers: int = 0) -> DataLoader:
    dataset = IQBurstDataset(data_dir, jitter=jitter)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers), dataset.cond_dim
