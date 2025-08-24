"""Sample or evaluate log-likelihood from a trained CNF."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .model import CNF
from .data import IQBurstDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample or evaluate CNF")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--mode", choices=["sample", "ll"], required=True)
    parser.add_argument("--num", type=int, default=16)
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--file", type=str, help="Input file for log-likelihood")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    cond_dim = ckpt.get("cond_dim", 1)
    channels = ckpt.get("channels", 2)
    hidden = ckpt.get("hidden", 64)
    blocks = ckpt.get("blocks", 4)
    model = CNF(channels, hidden, blocks, cond_dim).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    if args.mode == "sample":
        cond = torch.zeros(args.num, cond_dim, device=device)
        samples = model.sample(args.num, length=args.length, cond=cond)
        np.save("samples.npy", samples.cpu().numpy())
    else:
        assert args.file is not None, "--file required for ll mode"
        dataset = IQBurstDataset(Path(args.file).parent)
        arr = np.load(args.file)
        arr = np.stack([arr.real, arr.imag], axis=-1).astype(np.float32)
        cond = dataset.cond_for_path(args.file).unsqueeze(0)
        arr = torch.from_numpy(arr).unsqueeze(0).to(device)
        cond = cond.to(device)
        ll = -model.loss(arr, cond).item()
        print(f"log-likelihood: {ll}")


if __name__ == "__main__":
    main()
