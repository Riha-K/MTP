"""Train ResNet multi-label on MultiSenGE S1 (70% MD5 split).

Example (GPU machine):
  python -m baresoil.cnn.train \\
    --labels-dir data/baresoil_s1/ai4lcc/multisenge/labels \\
    --s1-dir data/baresoil_s1/ai4lcc/multisenge/s1 \\
    --out-dir checkpoints/resnet18_s1_v0.2 \\
    --arch resnet18 --epochs 20 --batch-size 64
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import S1MultiLabelDataset, build_train_index, collate_batch
from .model import build_resnet
from .split import DEFAULT_TRAIN_RATIO, NUM_CLASSES


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train ResNet multi-label on S1 VH.")
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--s1-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--arch", type=str, default="resnet18", choices=["resnet18", "resnet50"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    parser.add_argument("--max-samples", type=int, default=None, help="Smoke: limit train patches.")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    if not args.labels_dir.is_dir():
        print(f"Labels dir not found: {args.labels_dir}", file=sys.stderr)
        return 1
    if not args.s1_dir.is_dir():
        print(f"S1 dir not found: {args.s1_dir}", file=sys.stderr)
        return 1

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} arch={args.arch}")

    rows = build_train_index(
        labels_dir=args.labels_dir,
        s1_dir=args.s1_dir,
        train_ratio=args.train_ratio,
        max_samples=args.max_samples,
    )
    if not rows:
        print("No train patches resolved. Check labels/s1 paths.", file=sys.stderr)
        return 1
    print(f"train patches={len(rows)}")

    ds = S1MultiLabelDataset(rows)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_batch,
    )

    model = build_resnet(arch=args.arch, num_classes=NUM_CLASSES, pretrained=args.pretrained)
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n = 0
        pbar = tqdm(loader, desc=f"epoch {epoch}/{args.epochs}")
        for batch in pbar:
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * images.size(0)
            n += images.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        epoch_loss = running / max(n, 1)
        history.append({"epoch": epoch, "train_loss": epoch_loss})
        print(f"epoch {epoch}: train_loss={epoch_loss:.6f}")

        ckpt = {
            "arch": args.arch,
            "num_classes": NUM_CLASSES,
            "train_ratio": args.train_ratio,
            "epoch": epoch,
            "train_loss": epoch_loss,
            "model": model.state_dict(),
        }
        torch.save(ckpt, args.out_dir / "last.pt")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(ckpt, args.out_dir / "best.pt")

    meta = {
        "arch": args.arch,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "train_ratio": args.train_ratio,
        "num_train": len(rows),
        "best_train_loss": best_loss,
        "history": history,
        "out_dir": str(args.out_dir),
    }
    (args.out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved checkpoints -> {args.out_dir} (best.pt, last.pt)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
