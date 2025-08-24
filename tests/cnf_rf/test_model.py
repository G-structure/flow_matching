import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from cnf_rf.model import build_flow


def test_flow_forward_shapes():
    batch, channels, T = 2, 1, 4
    x = torch.randn(batch, channels, T, 2)
    cond = torch.zeros(batch, 0)
    model = build_flow(channels, cond_dim=0, num_layers=1)
    loss, zT = model(x, cond)
    assert zT.shape == x.shape
    assert loss.shape == ()
