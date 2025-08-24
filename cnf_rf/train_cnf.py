"""Training script for complex CNF."""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cnf_rf.data import IQDataset
from cnf_rf.model import build_flow
from flow_matching.train_utils import FlowMatchingTrainer


def parse_args():
    p = argparse.ArgumentParser(description="Train complex CNF")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--jitter", type=int, default=0)
    p.add_argument("--save", type=str, default="ckpt.pt")
    return p.parse_args()


def main():
    args = parse_args()
    ds = IQDataset(args.data_dir, max_jitter=args.jitter, cond_dim=0)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    sample, _ = ds[0]
    channels = sample.shape[0]
    model = build_flow(channels, cond_dim=0)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    trainer = FlowMatchingTrainer(model, opt, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    trainer.train(dl, epochs=args.epochs, save_path=args.save, save_every=10)


if __name__ == "__main__":
    main()
