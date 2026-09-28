"""Train CMU-ViT U-TAE Stage 2 (S1 ViT + S2 CNN, CONCAT bottleneck, S2 skips).

Phases:
  --mode head  = P4 (freeze S2 enc + S1 ViT + L-TAEs; train adapter/fusion/decoder)
  --mode full  = P5 (--init-ckpt from P4 best.pt)

Requires Stage 1 student weights:
  --student-ckpt multisenge_utae/checkpoints/cmu_s1_vit_v0/student_best.pt

Example smoke:
  python -m multisenge_utae.train_cmu_vit \\
    --index multisenge_seg/artifacts/patch_index.json \\
    --student-ckpt multisenge_utae/checkpoints/cmu_s1_vit_smoke/student_best.pt \\
    --num-classes 6 --mode head --epochs 1 \\
    --max-train 4 --max-val 2 \\
    --out-dir multisenge_utae/checkpoints/cmu_vit_c6_head_smoke
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
from multisenge_seg.taxonomy import num_output_classes
from multisenge_seg.train import (
    apply_class_boost,
    estimate_class_counts,
    estimate_class_counts_from_gr,
    parse_class_boost,
    set_seed,
    _seed_worker,
)
from multisenge_utae.data import batch_positions, collate_utae
from multisenge_utae.models.cmu_vit_utae import CMUViTUTAE


@torch.no_grad()
def evaluate(model: CMUViTUTAE, loader: DataLoader, device: torch.device, num_classes: int) -> dict:
  model.eval()
  cm = np.zeros((num_classes, num_classes), dtype=np.int64)
  for batch in loader:
    x = batch["x"].to(device)
    mask = batch["mask"].numpy()
    bp = batch_positions(x.shape[0], device)
    logits = model(x, batch_positions=bp)
    pred = logits.argmax(dim=1).cpu().numpy()
    cm += accumulate_confusion(pred, mask, num_classes=num_classes, ignore_index=255)
  return scores_from_cm(cm)


def train_one_epoch(
    model: CMUViTUTAE,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    accum_steps: int = 1,
) -> float:
  model.train()
  accum_steps = max(int(accum_steps), 1)
  total = 0.0
  n = 0
  opt.zero_grad(set_to_none=True)
  for i, batch in enumerate(loader):
    x = batch["x"].to(device)
    mask = batch["mask"].to(device)
    bp = batch_positions(x.shape[0], device)
    logits = model(x, batch_positions=bp)
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


def _load_torch(path: Path, device: torch.device):
  try:
    return torch.load(path, map_location=device, weights_only=False)
  except TypeError:
    return torch.load(path, map_location=device)


def norm_stats_from_ckpt(path: Path) -> dict:
  """Stats the S2 encoder and the CMU ViT were trained with. Do not re-estimate."""
  if not path.is_file():
    raise FileNotFoundError(f"checkpoint not found: {path}")
  ckpt = _load_torch(path, torch.device("cpu"))
  stats = ckpt.get("norm_stats") if isinstance(ckpt, dict) else None
  if not stats:
    side = path.parent / "norm_stats.json"
    if side.is_file():
      stats = json.loads(side.read_text(encoding="utf-8"))
  if not stats or "s2_mean" not in stats or "s1_mean" not in stats:
    raise RuntimeError(f"no norm_stats in {path} (and no sibling norm_stats.json)")
  return stats


def build_model(args, n_cls: int, device: torch.device) -> CMUViTUTAE:
  model = CMUViTUTAE(
      num_classes=n_cls,
      fusion=args.fusion,
      vit_image_size=args.vit_image_size,
  ).to(device)
  if args.student_ckpt is not None:
    model.load_student_vit(args.student_ckpt, map_location=device)
  if args.init_ckpt is None:
    if args.s2_ckpt is None or not args.s2_ckpt.is_file():
      raise SystemExit(f"missing S2 U-TAE checkpoint: {args.s2_ckpt}")
    model.load_s2_utae(args.s2_ckpt, map_location=device)
  if args.init_ckpt is not None:
    if not args.init_ckpt.is_file():
      raise FileNotFoundError(f"init checkpoint not found: {args.init_ckpt}")
    ckpt = _load_torch(args.init_ckpt, device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
      raise RuntimeError(
          f"init_ckpt mismatch missing={missing} unexpected={unexpected}"
      )
    print(f"loaded init_ckpt={args.init_ckpt}")
  model.set_train_mode(args.mode)
  trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
  total = sum(p.numel() for p in model.parameters())
  print(
      f"CMU-ViT-UTAE fusion={args.fusion} mode={args.mode} "
      f"trainable={trainable:,} / {total:,}"
  )
  return model


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
  meta = ckpt.get("args") or {}
  fusion = meta.get("fusion", args.fusion)

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
  loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=collate_utae)
  model = CMUViTUTAE(num_classes=n_cls, fusion=fusion, vit_image_size=args.vit_image_size).to(device)
  model.load_state_dict(ckpt["model"])
  print(f"eval ckpt={ckpt_path} split={args.eval_split} n={len(ds)} classes={n_cls}")
  scores = evaluate(model, loader, device, n_cls)
  out = args.out_dir
  out.mkdir(parents=True, exist_ok=True)
  dest = out / f"{args.eval_split}_metrics.json"
  dest.write_text(json.dumps(scores, indent=2), encoding="utf-8")
  print(
      f"{args.eval_split} wF1={scores['weighted_f1']:.4f} kappa={scores['kappa']:.4f} "
      f"acc={scores['accuracy']:.4f}"
  )
  print(format_prf_table(scores))
  print("wrote", dest)
  return 0


def main() -> int:
  p = argparse.ArgumentParser(description="Train CMU-ViT U-TAE Stage 2")
  p.add_argument("--data-root", type=Path, default=Path("LULCDial-s1/data/lulcdial_s1/ai4lcc/multisenge"))
  p.add_argument("--index", type=Path, default=None)
  p.add_argument("--num-classes", type=int, default=6, choices=[6, 10])
  p.add_argument("--mode", type=str, default="head", choices=["head", "full"])
  p.add_argument("--student-ckpt", type=Path, default=None, help="Stage 1 student_best.pt (load ViT)")
  p.add_argument(
      "--s2-ckpt",
      type=Path,
      default=Path("multisenge_utae/checkpoints/run_c10_s2_full_v0/best.pt"),
      help="10c S2-only U-TAE P5; encoder + L-TAE init for P4",
  )
  p.add_argument("--init-ckpt", type=Path, default=None, help="P4 best.pt for P5")
  p.add_argument("--fusion", type=str, default="concat", choices=["concat", "gated"])
  p.add_argument("--vit-image-size", type=int, default=256)
  p.add_argument("--epochs", type=int, default=80)
  p.add_argument("--batch-size", type=int, default=2)
  p.add_argument("--lr", type=float, default=1e-3)
  p.add_argument("--workers", type=int, default=2)
  p.add_argument("--patience", type=int, default=20)
  p.add_argument("--plateau-patience", type=int, default=5)
  p.add_argument("--no-augment", action="store_true")
  p.add_argument("--stats-patches", type=int, default=64)
  p.add_argument("--accum-steps", type=int, default=1)
  p.add_argument("--full-class-weights", action="store_true")
  p.add_argument("--seed", type=int, default=None)
  p.add_argument("--monitor", type=str, default="weighted_f1", choices=["weighted_f1", "kappa", "mean_f1"])
  p.add_argument("--class-boost", type=str, default="")
  p.add_argument("--max-train", type=int, default=None)
  p.add_argument("--max-val", type=int, default=None)
  p.add_argument("--out-dir", type=Path, default=Path("multisenge_utae/checkpoints/cmu_vit_c6_head_v0"))
  p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
  p.add_argument("--eval-ckpt", type=Path, default=None)
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

  if args.mode == "head" and args.student_ckpt is None:
    raise SystemExit("--student-ckpt required for P4 head (Stage 1 ViT weights)")
  if args.mode == "full" and args.init_ckpt is None:
    print("WARNING: P5 without --init-ckpt (random decoder); prefer P4 best.pt")

  if args.seed is not None:
    set_seed(args.seed)
    print("seed", args.seed)

  if args.mode == "full" and args.init_ckpt is not None:
    stats_src = args.init_ckpt
  else:
    stats_src = args.s2_ckpt
  print("using norm_stats from", stats_src)
  stats = norm_stats_from_ckpt(stats_src)

  train_ds = MultiSenGETemporalDataset(
      records, "train", num_classes=args.num_classes, augment=not args.no_augment,
      s1_mean=stats["s1_mean"], s1_std=stats["s1_std"],
      s2_mean=stats["s2_mean"], s2_std=stats["s2_std"],
  )
  val_ds = MultiSenGETemporalDataset(
      records, "val", num_classes=args.num_classes, augment=False,
      s1_mean=stats["s1_mean"], s1_std=stats["s1_std"],
      s2_mean=stats["s2_mean"], s2_std=stats["s2_std"],
  )
  if args.max_train:
    train_ds.records = train_ds.records[: args.max_train]
  if args.max_val:
    val_ds.records = val_ds.records[: args.max_val]

  loader_kw = dict(num_workers=args.workers, collate_fn=collate_utae)
  if args.seed is not None:
    g = torch.Generator()
    g.manual_seed(args.seed)
    loader_kw["generator"] = g
    loader_kw["worker_init_fn"] = _seed_worker
  train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kw)
  val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=collate_utae)

  print("train/val", len(train_ds), len(val_ds), "mode", args.mode, "fusion", args.fusion)
  device = torch.device(args.device)
  model = build_model(args, n_cls, device)

  if args.full_class_weights:
    counts = estimate_class_counts_from_gr(records, "train", args.num_classes)
  else:
    counts = estimate_class_counts(train_loader, n_cls, max_batches=80)
  print("counts", counts.tolist(), flush=True)
  weights = class_weights_from_counts(counts)
  boost = parse_class_boost(args.class_boost, n_cls)
  if np.any(boost != 1.0):
    weights = apply_class_boost(weights, boost)
  criterion = nn.CrossEntropyLoss(
      weight=torch.tensor(weights, dtype=torch.float32, device=device),
      ignore_index=255,
  )
  opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
      opt, mode="max", factor=0.1, patience=args.plateau_patience
  )

  args.out_dir.mkdir(parents=True, exist_ok=True)
  (args.out_dir / "norm_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
  history = []
  best_mon = -1.0
  stale = 0
  monitor_key = args.monitor
  for epoch in range(1, args.epochs + 1):
    t0 = time.time()
    print(f"epoch {epoch}/{args.epochs} train…", flush=True)
    loss = train_one_epoch(model, train_loader, opt, criterion, device, accum_steps=args.accum_steps)
    print(f"epoch {epoch}/{args.epochs} val…", flush=True)
    val = evaluate(model, val_loader, device, n_cls)
    mon = float(val[monitor_key])
    scheduler.step(mon)
    row = {
        "epoch": epoch,
        "train_loss": loss,
        "val_weighted_f1": val["weighted_f1"],
        "val_kappa": val["kappa"],
        "val_mean_f1": val["mean_f1"],
        "val_accuracy": val["accuracy"],
        "lr": opt.param_groups[0]["lr"],
        "mode": args.mode,
        "fusion": args.fusion,
        "sec": round(time.time() - t0, 1),
    }
    history.append(row)
    print(
        f"epoch {epoch:03d} loss={loss:.4f} val_wF1={val['weighted_f1']:.4f} "
        f"kappa={val['kappa']:.4f} mon={monitor_key}:{mon:.4f}"
    )
    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "num_classes": n_cls,
        "val": val,
        "norm_stats": stats,
        "mode": args.mode,
        "fusion": args.fusion,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    torch.save(ckpt, args.out_dir / "last.pt")
    if mon > best_mon:
      best_mon = mon
      stale = 0
      torch.save(ckpt, args.out_dir / "best.pt")
      (args.out_dir / "best_metrics.json").write_text(json.dumps(val, indent=2), encoding="utf-8")
    else:
      stale += 1
      if stale >= args.patience:
        print(f"EarlyStopping after {args.patience} epochs without val {monitor_key} improvement")
        break

  (args.out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
  print("done. best val", monitor_key, best_mon, "dir", args.out_dir)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
