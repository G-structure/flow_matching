import math
from typing import Callable, Optional

import torch
from torch import nn, Tensor
from torchdiffeq import odeint_adjoint as odeint


def divergence_from_func(func: Callable[[Tensor], Tensor], x: Tensor) -> Tensor:
    """Compute Hutchinson trace of the Jacobian of func at x."""
    noise = torch.randn_like(x)
    y = torch.sum(func(x) * noise)
    grad = torch.autograd.grad(y, x, create_graph=True)[0]
    return torch.sum(grad * noise, dim=list(range(1, grad.dim())))


class FlowMatchingWrapper(nn.Module):
    """Wrap a vector field f(x,t,cond) to compute CNF loss."""

    def __init__(self, forward_fn: Callable[[Tensor, Tensor, Optional[Tensor]], Tensor], t0: float = 0.0, t1: float = 1.0):
        super().__init__()
        self.forward_fn = forward_fn
        self.t0 = t0
        self.t1 = t1

    def _dynamics(self, t: Tensor, state: tuple, cond: Tensor):
        z, logp = state
        z.requires_grad_(True)
        f = self.forward_fn(z, t, cond)
        def func(inp):
            return self.forward_fn(inp, t, cond)
        div = divergence_from_func(func, z)
        return f, -div

    def forward(self, x: Tensor, cond: Optional[Tensor] = None, method: str = "rk4"):
        t = torch.tensor([self.t0, self.t1], device=x.device, dtype=torch.float32)
        logp0 = torch.zeros(x.shape[0], device=x.device, dtype=torch.float32)
        z_T, logp_T = odeint(lambda tt, yy: self._dynamics(tt, yy, cond), (x, logp0), t, method=method)
        z_T = z_T[-1]
        logp_T = logp_T[-1]
        base_const = 0.5 * x.shape[1] * x.shape[2] * math.log(2 * math.pi)
        nll = 0.5 * torch.sum(z_T ** 2, dim=[1, 2]) + base_const + logp_T
        return nll.mean(), z_T


class FlowMatchingTrainer:
    """Simple trainer utility for CNF models."""

    def __init__(self, model: FlowMatchingWrapper, optimizer: torch.optim.Optimizer, device: torch.device):
        self.model = model
        self.optimizer = optimizer
        self.device = device

    def train(self, dataloader: torch.utils.data.DataLoader, epochs: int, save_path: Optional[str] = None, save_every: int = 10):
        self.model.to(self.device)
        for epoch in range(1, epochs + 1):
            self.model.train()
            total = 0.0
            for x, cond in dataloader:
                x = x.to(self.device)
                cond = cond.to(self.device)
                loss, _ = self.model(x, cond)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total += loss.item() * x.size(0)
            print(f"Epoch {epoch}: loss={(total/len(dataloader.dataset)):.4f}")
            if save_path and epoch % save_every == 0:
                torch.save({'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}, save_path)
