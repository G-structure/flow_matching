"""Sampling or likelihood evaluation for complex CNF."""
import argparse
from pathlib import Path

import numpy as np
import torch

from cnf_rf.model import build_flow
from flow_matching.train_utils import FlowMatchingWrapper


def parse_args():
    p = argparse.ArgumentParser(description="Sample or evaluate CNF")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--mode", choices=["sample", "ll"], required=True)
    p.add_argument("--num", type=int, default=16)
    p.add_argument("--file", type=str)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    channels = ckpt['model']['forward_fn.layers.0.conv.conv_r.weight'].shape[1]
    model = build_flow(channels, cond_dim=0)
    model.load_state_dict(ckpt['model'])
    model.to(device)
    if args.mode == "sample":
        z = torch.randn(args.num, channels, ckpt.get('T', 64), 2, device=device)
        with torch.no_grad():
            _, samples = model(z, torch.zeros(args.num,0, device=device), method='euler')
        for i, s in enumerate(samples.cpu().numpy()):
            np.save(f"sample_{i}.npy", s.view(np.complex64))
    else:
        assert args.file, "--file required for likelihood"
        arr = np.load(args.file)
        x = torch.view_as_real(torch.from_numpy(arr)).unsqueeze(0).float().to(device)
        ll, _ = model(x, torch.zeros(1,0, device=device), method='rk4')
        print("Log likelihood:", -ll.item())


if __name__ == "__main__":
    main()
