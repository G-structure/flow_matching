"""Complex-valued CNF model."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from flow_matching.train_utils import FlowMatchingWrapper
from complex.complex_module import ComplexConv3d, ComplexGELU, NaiveComplexLayerNorm


class FiLM(nn.Module):
    def __init__(self, cond_dim: int, hidden: int):
        super().__init__()
        self.to_scale = nn.Linear(cond_dim, hidden)
        self.to_shift = nn.Linear(cond_dim, hidden)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale = self.to_scale(cond).view(cond.size(0), -1, 1, 1, 1, 1)
        shift = self.to_shift(cond).view(cond.size(0), -1, 1, 1, 1, 1)
        return scale * x + shift


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        self.conv = ComplexConv3d(channels, channels, kernel_size=1, padding=0)
        self.act = ComplexGELU()
        self.norm = NaiveComplexLayerNorm([channels])
        self.film = FiLM(cond_dim, channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # Lift to 5D for ComplexConv3d: [B,C,T,2] -> [B,C,T,1,1,2]
        h = x.unsqueeze(-1).unsqueeze(-1).permute(0, 1, 2, 4, 5, 3)
        h = self.conv(h)
        h = self.act(h)
        # LayerNorm expects channel as last dim.
        h = h.permute(0, 2, 3, 4, 1, 5)
        h = self.norm(h)
        h = h.permute(0, 4, 1, 2, 3, 5)
        h = self.film(h, cond)
        h = h.permute(0, 1, 2, 5, 3, 4).squeeze(-1).squeeze(-1)
        return x + h


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
    wrapper = FlowMatchingWrapper(vf, t0=0.0, t1=1.0)
    return wrapper
