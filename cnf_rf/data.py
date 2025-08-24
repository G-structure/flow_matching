"""Data utilities for complex IQ bursts."""
from __future__ import annotations

import glob
import os
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class IQDataset(Dataset):
    """Dataset loading complex64 npy files with optional time jitter."""

    def __init__(self, folder: str, max_jitter: int = 0, cond_dim: int = 0):
        self.files = sorted(glob.glob(os.path.join(folder, "*.npy")))
        self.max_jitter = max_jitter
        self.cond_dim = cond_dim

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        arr = np.load(self.files[idx])  # shape [C,T] complex64
        x = torch.view_as_real(torch.from_numpy(arr))  # [C,T,2]
        if self.max_jitter > 0:
            shift = np.random.randint(-self.max_jitter, self.max_jitter + 1)
            x = torch.roll(x, shifts=shift, dims=1)
        cond = torch.zeros(self.cond_dim, dtype=torch.float32)
        return x.float(), cond
