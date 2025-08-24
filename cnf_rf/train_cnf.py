"""Train the complex CNF on a folder of IQ bursts."""
from __future__ import annotations

import argparse

import torch
from torch.utils.tensorboard import SummaryWriter

from .data import make_dataloader
from .model import CNF


def main() -> None:
    parser = argparse.ArgumentParser(description="Train complex CNF")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--jitter", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--checkpoint", type=str, default="ckpt.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader, cond_dim = make_dataloader(args.data_dir, args.batch_size, args.jitter)

    example, _ = next(iter(loader))
    channels = example.shape[1]

    model = CNF(channels=channels, hidden=args.hidden, blocks=args.blocks, cond_dim=cond_dim).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    writer = SummaryWriter()

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        for x, cond in loader:
            x = x.to(device)
            cond = cond.to(device)
            loss = model.loss(x, cond)
            optim.zero_grad()
            loss.backward()
            optim.step()
            writer.add_scalar("loss", loss.item(), global_step)
            global_step += 1
        if epoch % args.save_every == 0:
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "cond_dim": cond_dim,
                    "channels": channels,
                    "hidden": args.hidden,
                    "blocks": args.blocks,
                },
                args.checkpoint,
            )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "cond_dim": cond_dim,
            "channels": channels,
            "hidden": args.hidden,
            "blocks": args.blocks,
        },
        args.checkpoint,
    )


if __name__ == "__main__":
    main()
