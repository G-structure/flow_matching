"""Complex-valued CNF model."""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint

from flow_matching.train_utils import FlowMatchingWrapper
from complex.complex_module import ComplexConv3d, ComplexGELU, NaiveComplexLayerNorm


class FiLM(nn.Module):
    def __init__(self, cond_dim: int, hidden: int):
        super().__init__()
        self.to_scale = nn.Linear(cond_dim, hidden)
        self.to_shift = nn.Linear(cond_dim, hidden)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale = self.to_scale(cond).unsqueeze(-1)
        shift = self.to_shift(cond).unsqueeze(-1)
        return scale[..., None] * x + shift[..., None]


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        self.conv = ComplexConv3d(channels, channels, kernel_size=1, padding=0)
        self.act = ComplexGELU()
        self.norm = NaiveComplexLayerNorm([channels])
        self.film = FiLM(cond_dim, channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        y = self.act(y)
        y = self.norm(y)
        y = self.film(y, cond)
        return x + y


class ComplexVectorField(nn.Module):
    def __init__(self, channels: int, cond_dim: int, num_layers: int = 4):
        super().__init__()
        self.layers = nn.ModuleList([
            ResidualBlock(channels, cond_dim) for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h, cond)
        return h


def build_flow(channels: int, cond_dim: int, num_layers: int = 4) -> FlowMatchingWrapper:
    vf = ComplexVectorField(channels, cond_dim, num_layers)
    wrapper = FlowMatchingWrapper(vf.forward, t0=0.0, t1=1.0)
    return wrapper
