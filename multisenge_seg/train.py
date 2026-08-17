"""Train ConvLSTM+Inception-S1S2 on MultiSenGE (CNN validation).

Intended for PARAM (needs torch + rasterio). Example:

  python -m multisenge_seg.build_index
  python -m multisenge_seg.train \\
    --index multisenge_seg/artifacts/patch_index.json \\
    --num-classes 6 --epochs 80 --batch-size 2 \\
    --out-dir multisenge_seg/checkpoints/run_c6_v0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from multisenge_seg.build_index import records_from_json
from multisenge_seg.dataset import MultiSenGETemporalDataset, PatchRecord, build_patch_index
from multisenge_seg.metrics import (
    accumulate_confusion,
    class_weights_from_counts,
    format_prf_table,
    scores_from_cm,
)
from multisenge_seg.model import ConvLSTMInceptionS1S2
from multisenge_seg.taxonomy import num_output_classes, remap_mask


def _collate(batch):
    return {
        "patch_id": [b["patch_id"] for b in batch],
        "s1": torch.stack([b["s1"] for b in batch], dim=0),
        "s2": torch.stack([b["s2"] for b in batch], dim=0),
        "mask": torch.stack([b["mask"] for b in batch], dim=0),
    }


@torch.no_grad()
def estimate_class_counts(loader, num_classes: int, max_batches: int = 50) -> np.ndarray:
    counts = np.zeros(num_classes, dtype=np.int64)
    for i, batch in enumerate(loader):
        m = batch["mask"].numpy()
        for c in range(num_classes):
            counts[c] += int((m == c).sum())
        if i + 1 >= max_batches:
            break
    return counts


def estimate_class_counts_from_gr(
    records: list[PatchRecord],
    split: str,
    num_classes: int,
) -> np.ndarray:
    """Inverse-frequency weights from all train GR masks (paper: full train set).

    Reads ground_reference only — not S1/S2 — so a full train scan is cheap.
    """
    import rasterio

    counts = np.zeros(num_classes, dtype=np.int64)
    n = 0
    for r in records:
        if r.split != split:
            continue
        with rasterio.open(r.gr_path) as src:
            mask = src.read(1).astype(np.int64)
        mask = remap_mask(mask, num_classes=num_classes)
        for c in range(1, num_classes + 1):
            counts[c - 1] += int((mask == c).sum())
        n += 1
    print("class-count patches", n)
    return counts


def estimate_channel_stats(
    records: list[PatchRecord],
    num_classes: int,
    max_patches: int = 64,
) -> dict:
    """Multitemporal per-channel mean/std on a train subset (paper-style)."""
    ds = MultiSenGETemporalDataset(records, "train", num_classes=num_classes, augment=False)
    n = len(ds) if max_patches <= 0 else min(len(ds), max_patches)
    if n == 0:
        raise RuntimeError("no train patches for stats")
    s1_sum = s1_sq = s2_sum = s2_sq = None
    pixels = 0
    # Read raw by temporarily bypassing normalize: use per-sample identity via fake mean0 std1 then undo — easier: load via dataset internals
    import rasterio

    for i in range(n):
        r = ds.records[i]
        s2_stack, s1_stack = [], []
        for m in ds.months:
            with rasterio.open(r.s2_by_month[m]) as src:
                s2_stack.append(src.read().astype(np.float32)[:10])
            with rasterio.open(r.s1_by_month[m]) as src:
                s1_stack.append(src.read().astype(np.float32)[:2])
        s2 = np.stack(s2_stack, 0)
        s1 = np.stack(s1_stack, 0)
        if s1_sum is None:
            s1_sum = np.zeros(s1.shape[1], np.float64)
            s1_sq = np.zeros(s1.shape[1], np.float64)
            s2_sum = np.zeros(s2.shape[1], np.float64)
            s2_sq = np.zeros(s2.shape[1], np.float64)
        s1_sum += s1.sum(axis=(0, 2, 3))
        s1_sq += (s1.astype(np.float64) ** 2).sum(axis=(0, 2, 3))
        s2_sum += s2.sum(axis=(0, 2, 3))
        s2_sq += (s2.astype(np.float64) ** 2).sum(axis=(0, 2, 3))
        pixels += s1.shape[0] * s1.shape[2] * s1.shape[3]
    s1_mean = s1_sum / pixels
    s2_mean = s2_sum / pixels
    s1_std = np.sqrt(np.maximum(s1_sq / pixels - s1_mean**2, 1e-12))
    s2_std = np.sqrt(np.maximum(s2_sq / pixels - s2_mean**2, 1e-12))
    return {
        "s1_mean": s1_mean.astype(np.float32).tolist(),
        "s1_std": s1_std.astype(np.float32).tolist(),
        "s2_mean": s2_mean.astype(np.float32).tolist(),
        "s2_std": s2_std.astype(np.float32).tolist(),
        "n_patches": n,
    }


@torch.no_grad()
def evaluate(model, loader, device, num_classes: int) -> dict:
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for batch in loader:
        s1 = batch["s1"].to(device)
        s2 = batch["s2"].to(device)
        mask = batch["mask"].numpy()
        logits = model(s1, s2)
        pred = logits.argmax(dim=1).cpu().numpy()
        cm += accumulate_confusion(pred, mask, num_classes=num_classes, ignore_index=255)
    return scores_from_cm(cm)


def _run_eval(args, records: list[PatchRecord], n_cls: int) -> int:
    ckpt_path = args.eval_ckpt
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    device = torch.device(args.device)
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    stats = ckpt.get("norm_stats")
    if not stats:
        stats_path = ckpt_path.parent / "norm_stats.json"
        if not stats_path.is_file():
            raise RuntimeError("checkpoint has no norm_stats; expected sibling norm_stats.json")
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
    n_cls = int(ckpt.get("num_classes", n_cls))
    s1_ch = int(ckpt.get("s1_ch", 2))
    s2_ch = int(ckpt.get("s2_ch", 10))
    ds = MultiSenGETemporalDataset(
        records,
        args.eval_split,
        num_classes=6 if n_cls == 6 else 10,
        augment=False,
        s1_mean=stats["s1_mean"],
        s1_std=stats["s1_std"],
        s2_mean=stats["s2_mean"],
        s2_std=stats["s2_std"],
    )
    if len(ds) == 0:
        raise RuntimeError(f"no patches for split={args.eval_split}")
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=_collate,
    )
    model = ConvLSTMInceptionS1S2(s1_ch=s1_ch, s2_ch=s2_ch, num_classes=n_cls).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"eval ckpt={ckpt_path} split={args.eval_split} n={len(ds)} classes={n_cls} device={device}")
    scores = evaluate(model, loader, device, n_cls)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"{args.eval_split}_metrics.json"
    dest.write_text(json.dumps(scores, indent=2), encoding="utf-8")
    print(
        f"{args.eval_split} wP={scores['weighted_precision']:.4f} "
        f"wR={scores['weighted_recall']:.4f} wF1={scores['weighted_f1']:.4f} "
        f"acc={scores['accuracy']:.4f} kappa={scores['kappa']:.4f}"
    )
    print(format_prf_table(scores))
    print("wrote", dest)
    return 0


def train_one_epoch(model, loader, opt, criterion, device, accum_steps: int = 1) -> float:
    """Paper batch 16 ≈ batch_size × accum_steps when one GPU cannot hold 16."""
    model.train()
    accum_steps = max(int(accum_steps), 1)
    total = 0.0
    n = 0
    opt.zero_grad(set_to_none=True)
    for i, batch in enumerate(loader):
        s1 = batch["s1"].to(device)
        s2 = batch["s2"].to(device)
        mask = batch["mask"].to(device)
        logits = model(s1, s2)
        loss = criterion(logits, mask) / accum_steps
        loss.backward()
        if (i + 1) % accum_steps == 0:
            opt.step()
            opt.zero_grad(set_to_none=True)
        total += float(loss.item()) * accum_steps
        n += 1
    if n % accum_steps != 0:
        opt.step()
        opt.zero_grad(set_to_none=True)
    return total / max(n, 1)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path("LULCDial-s1/data/lulcdial_s1/ai4lcc/multisenge"))
    p.add_argument("--index", type=Path, default=None, help="Optional cached patch_index JSON")
    p.add_argument("--num-classes", type=int, default=6, choices=[6, 10])
    p.add_argument("--epochs", type=int, default=80, help="Max epochs; EarlyStopping may stop sooner")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--patience", type=int, default=20, help="EarlyStopping patience (paper)")
    p.add_argument("--plateau-patience", type=int, default=5, help="ReduceLROnPlateau patience")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument(
        "--stats-patches",
        type=int,
        default=64,
        help="Train patches for channel mean/std. 0 = all train (paper).",
    )
    p.add_argument(
        "--accum-steps",
        type=int,
        default=1,
        help="Gradient accumulation. Effective batch = batch-size × accum-steps (paper 16).",
    )
    p.add_argument("--full-class-weights", action="store_true", help="Count class pixels on all train GR masks")
    p.add_argument("--max-train", type=int, default=None, help="Smoke: limit train patches")
    p.add_argument("--max-val", type=int, default=None)
    p.add_argument("--out-dir", type=Path, default=Path("multisenge_seg/checkpoints/run_c6_v0"))
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--eval-ckpt",
        type=Path,
        default=None,
        help="Skip train; load this checkpoint and score --eval-split (paper test = 31UEQ)",
    )
    p.add_argument("--eval-split", type=str, default="test", choices=["train", "val", "test"])
    args = p.parse_args()

    n_cls = num_output_classes(args.num_classes)
    if args.index and args.index.is_file():
        records = records_from_json(args.index)
        print("loaded index", args.index, "n=", len(records))
    else:
        print("building index from", args.data_root)
        records = build_patch_index(args.data_root)

    if args.eval_ckpt:
        return _run_eval(args, records, n_cls)

    print("estimating channel stats…")
    stats = estimate_channel_stats(records, args.num_classes, max_patches=args.stats_patches)
    print("stats patches", stats["n_patches"])

    train_ds = MultiSenGETemporalDataset(
        records,
        "train",
        num_classes=args.num_classes,
        augment=not args.no_augment,
        s1_mean=stats["s1_mean"],
        s1_std=stats["s1_std"],
        s2_mean=stats["s2_mean"],
        s2_std=stats["s2_std"],
    )
    val_ds = MultiSenGETemporalDataset(
        records,
        "val",
        num_classes=args.num_classes,
        augment=False,
        s1_mean=stats["s1_mean"],
        s1_std=stats["s1_std"],
        s2_mean=stats["s2_mean"],
        s2_std=stats["s2_std"],
    )
    if args.max_train:
        train_ds.records = train_ds.records[: args.max_train]
    if args.max_val:
        val_ds.records = val_ds.records[: args.max_val]

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=_collate,
    )

    sample = train_ds[0]
    s1_ch = int(sample["s1"].shape[1])
    s2_ch = int(sample["s2"].shape[1])
    print("channels s1/s2", s1_ch, s2_ch, "train/val", len(train_ds), len(val_ds))

    device = torch.device(args.device)
    model = ConvLSTMInceptionS1S2(s1_ch=s1_ch, s2_ch=s2_ch, num_classes=n_cls).to(device)

    if args.full_class_weights:
        print("estimating class weights (all train GR masks)…")
        counts = estimate_class_counts_from_gr(records, "train", args.num_classes)
    else:
        print("estimating class weights (subset of train)…")
        counts = estimate_class_counts(train_loader, n_cls, max_batches=80)
    weights = class_weights_from_counts(counts)
    print("counts", counts.tolist(), "weights", weights.round(4).tolist())
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device),
        ignore_index=255,
    )
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.1, patience=args.plateau_patience
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "norm_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    history = []
    best_f1 = -1.0
    stale = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss = train_one_epoch(
            model, train_loader, opt, criterion, device, accum_steps=args.accum_steps
        )
        val = evaluate(model, val_loader, device, n_cls)
        scheduler.step(val["weighted_f1"])
        row = {
            "epoch": epoch,
            "train_loss": loss,
            "val_weighted_f1": val["weighted_f1"],
            "val_accuracy": val["accuracy"],
            "val_kappa": val["kappa"],
            "val_mean_f1": val["mean_f1"],
            "lr": opt.param_groups[0]["lr"],
            "sec": round(time.time() - t0, 1),
        }
        history.append(row)
        print(
            f"epoch {epoch:03d} loss={loss:.4f} "
            f"val_wF1={val['weighted_f1']:.4f} acc={val['accuracy']:.4f} "
            f"kappa={val['kappa']:.4f} lr={row['lr']:.1e} ({row['sec']}s)"
        )
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "num_classes": n_cls,
            "s1_ch": s1_ch,
            "s2_ch": s2_ch,
            "val": val,
            "norm_stats": stats,
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        }
        torch.save(ckpt, args.out_dir / "last.pt")
        if val["weighted_f1"] > best_f1:
            best_f1 = val["weighted_f1"]
            stale = 0
            torch.save(ckpt, args.out_dir / "best.pt")
            (args.out_dir / "best_metrics.json").write_text(json.dumps(val, indent=2), encoding="utf-8")
        else:
            stale += 1
            if stale >= args.patience:
                print(f"EarlyStopping after {args.patience} epochs without val_wF1 improvement")
                break

    (args.out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print("done. best val weighted F1", best_f1, "dir", args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
