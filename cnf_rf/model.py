"""Complex-valued CNF model for RF IQ bursts."""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint

from complex.complex_module import (
    ComplexGELU,
    NaiveComplexLayerNorm,
    complex_mul,
)


class ComplexConv1d(nn.Module):
    """Simple complex 1D convolution."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 1):
        super().__init__()
        self.conv_r = nn.Conv1d(in_ch, out_ch, kernel_size)
        self.conv_i = nn.Conv1d(in_ch, out_ch, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xr, xi = x.unbind(dim=-1)
        yr = self.conv_r(xr) - self.conv_i(xi)
        yi = self.conv_r(xi) + self.conv_i(xr)
        return torch.stack((yr, yi), dim=-1)


class FiLM(nn.Module):
    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.linear = nn.Linear(cond_dim, channels * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale_shift = self.linear(cond).unsqueeze(-1)
        scale, shift = scale_shift.chunk(2, dim=1)
        xr, xi = x.unbind(dim=-1)
        xr = xr * scale + shift
        xi = xi * scale + shift
        return torch.stack((xr, xi), dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        self.conv = ComplexConv1d(channels, channels, kernel_size=1)
        self.act = ComplexGELU()
        self.norm = NaiveComplexLayerNorm(channels)
        self.film = FiLM(cond_dim, channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        y = self.act(y)
        y = y.permute(0, 2, 1, 3)
        y = self.norm(y)
        y = y.permute(0, 2, 1, 3)
        y = self.film(y, cond)
        return x + y


class ComplexVectorField(nn.Module):
    def __init__(self, channels: int, hidden: int, blocks: int, cond_dim: int):
        super().__init__()
        self.init = ComplexConv1d(channels, hidden, kernel_size=1)
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden, cond_dim) for _ in range(blocks)]
        )
        self.out = ComplexConv1d(hidden, channels, kernel_size=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.init(x)
        for blk in self.blocks:
            h = blk(h, cond)
        h = self.out(h)
        return h


class ODEFunc(nn.Module):
    def __init__(self, vf: ComplexVectorField, cond_dim: int):
        super().__init__()
        self.vf = vf
        self.cond_dim = cond_dim
        self.cond: torch.Tensor | None = None

    def set_condition(self, cond: torch.Tensor):
        self.cond = cond

    def forward(self, t: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor]):
        z, logp = states
        cond = self.cond
        assert cond is not None
        if torch.is_grad_enabled():
            z = z.requires_grad_()
            drift = self.vf(z, t, cond)
            div = torch.autograd.grad(
                drift,
                z,
                torch.ones_like(drift),
                create_graph=True,
            )[0]
            div = div.reshape(z.shape[0], -1).sum(dim=1)
        else:
            drift = self.vf(z, t, cond)
            div = torch.zeros(z.shape[0], device=z.device)
        return drift, -div


class CNF(nn.Module):
    def __init__(self, channels: int, hidden: int, blocks: int, cond_dim: int):
        super().__init__()
        self.channels = channels
        self.vf = ComplexVectorField(channels, hidden, blocks, cond_dim)
        self.odefunc = ODEFunc(self.vf, cond_dim)
        self.t0, self.t1 = 0.0, 1.0

    def loss(self, x: torch.Tensor, cond: torch.Tensor, method: str = "rk4") -> torch.Tensor:
        logp = torch.zeros(x.shape[0], device=x.device)
        self.odefunc.set_condition(cond)
        t = torch.tensor([self.t0, self.t1], device=x.device)
        z_T, logp_T = odeint(self.odefunc, (x, logp), t, method=method)
        z_T, logp_T = z_T[-1], logp_T[-1]
        const = 0.5 * math.log(2 * math.pi)
        dim = z_T.shape[1] * z_T.shape[2] * 2
        base_logprob = -0.5 * (z_T ** 2).view(z_T.shape[0], -1).sum(dim=1) - dim * const
        nll = -(base_logprob + logp_T)
        return nll.mean()

    @torch.no_grad()
    def sample(self, num: int, length: int, cond: torch.Tensor, method: str = "euler") -> torch.Tensor:
        z = torch.randn(num, self.channels, length, 2, device=cond.device)
        logp = torch.zeros(num, device=cond.device)
        self.odefunc.set_condition(cond)
        t = torch.tensor([self.t1, self.t0], device=cond.device)
        x, _ = odeint(self.odefunc, (z, logp), t, method=method)
        return x[-1]
