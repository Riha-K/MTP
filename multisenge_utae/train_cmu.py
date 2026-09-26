"""Stage 1 CMU: align S1 ViT-B/16 to frozen MultiSenGE S2 U-TAE encoder.

Locked plan (§8):
  teacher = run_c10_s2_full_v0 encoder (spatial only, no L-TAE)
  student = ViT-B/16 in_chans=2
  pairs = same patch, same date t (flatten B*T)
  InfoNCE proj=256, tau=0.07

Example smoke:
  python -m multisenge_utae.train_cmu \\
    --index multisenge_seg/artifacts/patch_index.json \\
    --teacher-ckpt multisenge_utae/checkpoints/run_c10_s2_full_v0/best.pt \\
    --epochs 1 --max-train 4 --max-val 2 \\
    --out-dir multisenge_utae/checkpoints/cmu_s1_vit_smoke
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from multisenge_seg.build_index import records_from_json
from multisenge_seg.dataset import MultiSenGETemporalDataset, build_patch_index
from multisenge_seg.train import set_seed, _seed_worker
from multisenge_utae.data import collate_utae
from multisenge_utae.models.cmu import ProjHead, global_pool, info_nce, retrieval_acc
from multisenge_utae.models.s1_vit import S1ViTB16
from multisenge_utae.models.utae import UTAE


def _load_ckpt(path: Path, device: torch.device) -> dict:
  if not path.is_file():
    raise FileNotFoundError(f"checkpoint not found: {path}")
  try:
    return torch.load(path, map_location=device, weights_only=False)
  except TypeError:
    return torch.load(path, map_location=device)


def load_frozen_s2_teacher(ckpt_path: Path, device: torch.device) -> tuple[UTAE, dict]:
  ckpt = _load_ckpt(ckpt_path, device)
  num_classes = int(ckpt.get("num_classes", 10))
  input_dim = int(ckpt.get("input_dim", 10))
  teacher = UTAE(input_dim=input_dim, num_classes=num_classes).to(device)
  state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
  missing, unexpected = teacher.load_state_dict(state, strict=False)
  print(f"teacher ckpt={ckpt_path} input_dim={input_dim} missing={len(missing)} unexpected={len(unexpected)}")
  teacher.eval()
  for p in teacher.parameters():
    p.requires_grad = False
  stats = ckpt.get("norm_stats")
  if not stats:
    stats_path = ckpt_path.parent / "norm_stats.json"
    if stats_path.is_file():
      stats = json.loads(stats_path.read_text(encoding="utf-8"))
  if not stats:
    raise RuntimeError("teacher checkpoint has no norm_stats (and no sibling norm_stats.json)")
  return teacher, stats


def flatten_dates(x: torch.Tensor) -> torch.Tensor:
  """B,T,C,H,W → (B*T),C,H,W"""
  b, t, c, h, w = x.shape
  return x.reshape(b * t, c, h, w)


def run_epoch(
    teacher: UTAE,
    student: S1ViTB16,
    proj_t: ProjHead,
    proj_s: ProjHead,
    loader: DataLoader,
    opt: torch.optim.Optimizer | None,
    device: torch.device,
    temperature: float,
    train: bool,
) -> dict:
  student.train(train)
  proj_s.train(train)
  proj_t.train(train)
  total_loss = 0.0
  total_acc = 0.0
  n = 0
  for batch in loader:
    s2 = batch["s2"].to(device)  # B,T,10,H,W
    s1 = batch["s1"].to(device)  # B,T,2,H,W
    s2_f = flatten_dates(s2)
    s1_f = flatten_dates(s1)

    with torch.set_grad_enabled(train):
      with torch.no_grad():
        # teacher encoder always frozen; projector may train
        feat_t = teacher.encode_spatial_bottleneck(s2_f.unsqueeze(1))
        pooled_t = global_pool(feat_t)
      z_t = proj_t(pooled_t)
      z_s = proj_s(student(s1_f))
      # stopgrad on teacher side of contrastive (encoder already frozen)
      loss = info_nce(z_s, z_t.detach(), temperature=temperature)

      if train:
        assert opt is not None
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    acc = retrieval_acc(z_s.detach(), z_t.detach())
    total_loss += float(loss.item())
    total_acc += acc
    n += 1

  return {"loss": total_loss / max(n, 1), "acc": total_acc / max(n, 1)}


def main() -> int:
  p = argparse.ArgumentParser(description="Stage 1 CMU: S1 ViT ↔ frozen S2 U-TAE")
  p.add_argument("--data-root", type=Path, default=Path("LULCDial-s1/data/lulcdial_s1/ai4lcc/multisenge"))
  p.add_argument("--index", type=Path, default=None)
  p.add_argument(
      "--teacher-ckpt",
      type=Path,
      default=Path("multisenge_utae/checkpoints/run_c10_s2_full_v0/best.pt"),
  )
  p.add_argument("--epochs", type=int, default=40)
  p.add_argument("--batch-size", type=int, default=2, help="patch batch; effective InfoNCE N = batch*T")
  p.add_argument("--lr", type=float, default=3e-4)
  p.add_argument("--workers", type=int, default=2)
  p.add_argument("--patience", type=int, default=12)
  p.add_argument("--temperature", type=float, default=0.07)
  p.add_argument("--proj-dim", type=int, default=256)
  p.add_argument("--image-size", type=int, default=256)
  p.add_argument("--no-augment", action="store_true")
  p.add_argument("--stats-patches", type=int, default=64)
  p.add_argument("--max-train", type=int, default=None)
  p.add_argument("--max-val", type=int, default=None)
  p.add_argument("--seed", type=int, default=42)
  p.add_argument("--out-dir", type=Path, default=Path("multisenge_utae/checkpoints/cmu_s1_vit_v0"))
  p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
  args = p.parse_args()

  if args.seed is not None:
    set_seed(args.seed)

  if args.index and args.index.is_file():
    records = records_from_json(args.index)
    print("loaded index", args.index, "n=", len(records))
  else:
    print("building index from", args.data_root)
    records = build_patch_index(args.data_root)

  device = torch.device(args.device)
  teacher, stats = load_frozen_s2_teacher(args.teacher_ckpt, device)

  # Prefer teacher norm_stats so CMU matches the S2 P5 feature space.
  print("using teacher norm_stats")
  train_ds = MultiSenGETemporalDataset(
      records,
      "train",
      num_classes=10,
      augment=not args.no_augment,
      s1_mean=stats["s1_mean"],
      s1_std=stats["s1_std"],
      s2_mean=stats["s2_mean"],
      s2_std=stats["s2_std"],
  )
  val_ds = MultiSenGETemporalDataset(
      records,
      "val",
      num_classes=10,
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

  loader_kw = dict(num_workers=args.workers, collate_fn=collate_utae)
  if args.seed is not None:
    g = torch.Generator()
    g.manual_seed(args.seed)
    loader_kw["generator"] = g
    loader_kw["worker_init_fn"] = _seed_worker

  train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kw)
  val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=collate_utae)

  student = S1ViTB16(in_chans=2, image_size=args.image_size).to(device)
  bottleneck_dim = int(teacher.encoder_widths[-1])
  proj_t = ProjHead(bottleneck_dim, proj_dim=args.proj_dim).to(device)
  proj_s = ProjHead(student.embed_dim, proj_dim=args.proj_dim).to(device)

  params = list(student.parameters()) + list(proj_s.parameters()) + list(proj_t.parameters())
  opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.05)
  sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))

  n_train = sum(p.numel() for p in params)
  print(
      f"CMU student+proj params={n_train:,} train/val={len(train_ds)}/{len(val_ds)} "
      f"T=4 → InfoNCE N≈batch*{4} tau={args.temperature} proj={args.proj_dim}"
  )

  args.out_dir.mkdir(parents=True, exist_ok=True)
  (args.out_dir / "norm_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
  hparams = {
      "temperature": args.temperature,
      "proj_dim": args.proj_dim,
      "lr": args.lr,
      "batch_size": args.batch_size,
      "image_size": args.image_size,
      "teacher_ckpt": str(args.teacher_ckpt),
      "pairing": "same_patch_same_t",
      "student": "ViT-B/16 in_chans=2",
  }
  (args.out_dir / "cmu_hparams.json").write_text(json.dumps(hparams, indent=2), encoding="utf-8")

  history = []
  best_acc = -1.0
  stale = 0
  for epoch in range(1, args.epochs + 1):
    t0 = time.time()
    print(f"epoch {epoch}/{args.epochs} train…", flush=True)
    tr = run_epoch(teacher, student, proj_t, proj_s, train_loader, opt, device, args.temperature, train=True)
    print(f"epoch {epoch}/{args.epochs} val…", flush=True)
    va = run_epoch(teacher, student, proj_t, proj_s, val_loader, None, device, args.temperature, train=False)
    sched.step()
    row = {
        "epoch": epoch,
        "train_loss": tr["loss"],
        "train_acc": tr["acc"],
        "val_loss": va["loss"],
        "val_acc": va["acc"],
        "lr": opt.param_groups[0]["lr"],
        "sec": round(time.time() - t0, 1),
    }
    history.append(row)
    print(
        f"epoch {epoch:03d} loss={tr['loss']:.4f}/{va['loss']:.4f} "
        f"acc={tr['acc']:.3f}/{va['acc']:.3f} lr={row['lr']:.1e} sec={row['sec']}"
    )

    ckpt = {
        "epoch": epoch,
        "student": student.state_dict(),
        "proj_student": proj_s.state_dict(),
        "proj_teacher": proj_t.state_dict(),
        "teacher_ckpt": str(args.teacher_ckpt),
        "norm_stats": stats,
        "hparams": hparams,
        "val": va,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    torch.save(ckpt, args.out_dir / "last.pt")
    # Also save student-only weights for Stage 2 init
    torch.save({"model": student.state_dict(), "embed_dim": student.embed_dim}, args.out_dir / "student_last.pt")

    if va["acc"] > best_acc:
      best_acc = va["acc"]
      stale = 0
      torch.save(ckpt, args.out_dir / "best.pt")
      torch.save({"model": student.state_dict(), "embed_dim": student.embed_dim}, args.out_dir / "student_best.pt")
      (args.out_dir / "best_metrics.json").write_text(json.dumps(va, indent=2), encoding="utf-8")
    else:
      stale += 1
      if stale >= args.patience:
        print(f"EarlyStopping after {args.patience} epochs without val retrieval acc improvement")
        break

  (args.out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
  print("done. best val retrieval acc", best_acc, "dir", args.out_dir)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
