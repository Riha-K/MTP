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
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from multisenge_seg.build_index import records_from_json
from multisenge_seg.dataset import MultiSenGETemporalDataset, build_patch_index
from multisenge_seg.train import set_seed, _seed_worker
from multisenge_utae.data import collate_utae
from multisenge_utae.models.cmu import ProjHead, project_map, spatial_info_nce, spatial_retrieval_acc
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
  meta = ckpt.get("args") or {}
  activation = str(meta.get("activation", "relu"))
  teacher = UTAE(input_dim=input_dim, num_classes=num_classes, activation=activation).to(device)
  state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
  missing, unexpected = teacher.load_state_dict(state, strict=False)
  print(
      f"teacher ckpt={ckpt_path} input_dim={input_dim} activation={activation} "
      f"missing={len(missing)} unexpected={len(unexpected)}"
  )
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


class TeacherNegBank:
  """FIFO of detached teacher maps used only as InfoNCE negatives."""

  def __init__(self, capacity: int):
    self.capacity = int(capacity)
    self.slots: torch.Tensor | None = None
    self.filled = 0
    self.ptr = 0

  def get(self) -> torch.Tensor | None:
    if self.slots is None or self.filled == 0:
      return None
    if self.filled < self.capacity:
      return self.slots[: self.filled]
    return self.slots

  def add(self, z_t: torch.Tensor) -> None:
    z = z_t.detach()
    n, length, dim = z.shape
    if self.capacity <= 0 or n == 0:
      return
    if self.slots is None:
      self.slots = torch.empty(self.capacity, length, dim, device=z.device, dtype=z.dtype)
    elif self.slots.shape[1:] != (length, dim):
      raise RuntimeError(f"neg bank {tuple(self.slots.shape)} cannot store {tuple(z.shape)}")
    for i in range(n):
      self.slots[self.ptr].copy_(z[i])
      self.ptr = (self.ptr + 1) % self.capacity
      self.filled = min(self.capacity, self.filled + 1)


def _pair_maps(teacher, student, proj_t, proj_s, s2, s1):
  s2_f = flatten_dates(s2)
  s1_f = flatten_dates(s1)
  with torch.no_grad():
    feat_t = teacher.encode_spatial_bottleneck(s2_f.unsqueeze(1))
    if feat_t.dim() == 5:
      feat_t = feat_t.squeeze(1)
  tok_s = student.forward_tokens(s1_f)
  if feat_t.shape[-2:] != tok_s.shape[-2:]:
    feat_t = F.interpolate(feat_t, size=tok_s.shape[-2:], mode="bilinear", align_corners=False)
  z_t = project_map(proj_t, feat_t)
  z_s = project_map(proj_s, tok_s)
  return z_t, z_s


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
    bank: TeacherNegBank | None = None,
) -> dict:
  student.train(train)
  proj_s.train(train)
  proj_t.train(train)
  total_loss = 0.0
  total_acc = 0.0
  total_acc_k = 0.0
  n = 0
  for batch in loader:
    s2 = batch["s2"].to(device)  # B,T,10,H,W
    s1 = batch["s1"].to(device)  # B,T,2,H,W
    neg = None if bank is None else bank.get()

    with torch.set_grad_enabled(train):
      z_t, z_s = _pair_maps(teacher, student, proj_t, proj_s, s2, s1)
      loss = spatial_info_nce(z_s, z_t, temperature=temperature, neg_bank=neg)

      if train:
        assert opt is not None
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if bank is not None:
          bank.add(z_t)

    acc = spatial_retrieval_acc(z_s.detach(), z_t.detach())
    acc_k = spatial_retrieval_acc(z_s.detach(), z_t.detach(), neg_bank=neg)
    total_loss += float(loss.item())
    total_acc += acc
    total_acc_k += acc_k
    n += 1

  return {
      "loss": total_loss / max(n, 1),
      "acc": total_acc / max(n, 1),
      "acc_k": total_acc_k / max(n, 1),
  }


def main() -> int:
  p = argparse.ArgumentParser(description="Stage 1 CMU: S1 ViT ↔ frozen S2 U-TAE")
  p.add_argument("--data-root", type=Path, default=Path("LULCDial-s1/data/lulcdial_s1/ai4lcc/multisenge"))
  p.add_argument("--index", type=Path, default=None)
  p.add_argument(
      "--teacher-ckpt",
      type=Path,
      default=Path("multisenge_utae/checkpoints/run_c10_s2_full_v0/best.pt"),
  )
  p.add_argument("--epochs", type=int, default=80)
  p.add_argument("--batch-size", type=int, default=2, help="patch batch; effective InfoNCE N = batch*T")
  p.add_argument(
      "--neg-bank",
      type=int,
      default=0,
      help="extra detached teacher samples used only as InfoNCE negatives. 0 keeps N=batch*T.",
  )
  p.add_argument("--lr", type=float, default=3e-4)
  p.add_argument("--workers", type=int, default=2)
  p.add_argument("--patience", type=int, default=20)
  p.add_argument("--temperature", type=float, default=0.07)
  p.add_argument("--proj-dim", type=int, default=256)
  p.add_argument("--image-size", type=int, default=256)
  p.add_argument("--no-augment", action="store_true")
  p.add_argument("--stats-patches", type=int, default=64)
  p.add_argument("--max-train", type=int, default=None)
  p.add_argument("--max-val", type=int, default=None)
  p.add_argument("--seed", type=int, default=42)
  p.add_argument("--out-dir", type=Path, default=Path("multisenge_utae/checkpoints/cmu_s1_vit_v0"))
  p.add_argument(
      "--resume",
      type=Path,
      default=None,
      help="last.pt from a finished run. Loads ViT and both projectors, then trains start_epoch..--epochs with a new cosine.",
  )
  p.add_argument(
      "--fail-if-train-loss-above",
      type=float,
      default=None,
      help="exit 1 when the last train loss stays at chance (smoke gate)",
  )
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

  resume_ckpt = _load_ckpt(args.resume, device) if args.resume else None
  student = S1ViTB16(in_chans=2, image_size=args.image_size, pretrained=resume_ckpt is None).to(device)
  bottleneck_dim = int(teacher.encoder_widths[-1])
  proj_t = ProjHead(bottleneck_dim, proj_dim=args.proj_dim).to(device)
  proj_s = ProjHead(student.embed_dim, proj_dim=args.proj_dim).to(device)
  start_epoch = 1
  if resume_ckpt is not None:
    student.load_state_dict(resume_ckpt["student"])
    proj_s.load_state_dict(resume_ckpt["proj_student"])
    proj_t.load_state_dict(resume_ckpt["proj_teacher"])
    start_epoch = int(resume_ckpt["epoch"]) + 1
    if start_epoch > args.epochs:
      raise SystemExit(f"resume epoch {start_epoch - 1} is already past --epochs {args.epochs}")

  params = list(student.parameters()) + list(proj_s.parameters()) + list(proj_t.parameters())
  opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.05)
  cosine_epochs = args.epochs if resume_ckpt is None else (args.epochs - start_epoch + 1)
  sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cosine_epochs, 1))

  n_train = sum(p.numel() for p in params)
  bank = TeacherNegBank(args.neg_bank) if args.neg_bank > 0 else None
  if bank is not None:
    proj_t.eval()
    student.eval()
    seen = 0
    for batch in train_loader:
      if bank.filled >= bank.capacity:
        break
      s2 = batch["s2"].to(device)
      s1 = batch["s1"].to(device)
      with torch.no_grad():
        z_t, _z_s = _pair_maps(teacher, student, proj_t, proj_s, s2, s1)
      bank.add(z_t)
      seen += int(z_t.shape[0])
    n_cur = args.batch_size * 4
    keys = n_cur + bank.filled
    print(
        f"neg bank filled {bank.filled}/{bank.capacity} from {seen} train samples. "
        f"keys={keys} chance_acc={1.0 / keys:.4f} chance_loss={math.log(keys):.4f}"
    )
    print(f"best.pt follows val acc among the current {n_cur} samples (acc=). accK= includes the bank.")
  print(
      f"CMU student+proj params={n_train:,} train/val={len(train_ds)}/{len(val_ds)} "
      f"T=4 spatial InfoNCE over the batch at each ViT site "
      f"tau={args.temperature} proj={args.proj_dim} projector=Linear-LayerNorm-GELU-Linear "
      f"neg_bank={args.neg_bank}"
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
      "pairing": "same_patch_same_t_spatial",
      "projector": "Linear-LayerNorm-GELU-Linear",
      "student": "ViT-B/16 in_chans=2 ImageNet stem=mean RGB",
      "resume": str(args.resume) if args.resume else None,
      "start_epoch": start_epoch,
      "neg_bank": args.neg_bank,
      "info_nce_keys": args.batch_size * 4 + args.neg_bank,
  }
  (args.out_dir / "cmu_hparams.json").write_text(json.dumps(hparams, indent=2), encoding="utf-8")

  history: list[dict] = []
  best_acc = -1.0
  if resume_ckpt is not None:
    hist_path = args.out_dir / "history.json"
    if hist_path.is_file():
      history = [row for row in json.loads(hist_path.read_text(encoding="utf-8")) if int(row["epoch"]) < start_epoch]
    best_acc = max((float(row["val_acc"]) for row in history), default=-1.0)
    metrics_path = args.out_dir / "best_metrics.json"
    if metrics_path.is_file():
      best_acc = max(best_acc, float(json.loads(metrics_path.read_text(encoding="utf-8"))["acc"]))
  stale = 0
  if resume_ckpt is not None:
    print(
        f"resume weights from epoch {start_epoch - 1} -> epochs {start_epoch}..{args.epochs} "
        f"new cosine T_max={cosine_epochs} lr={args.lr:.1e} kept best val acc={best_acc:.3f}"
    )
  for epoch in range(start_epoch, args.epochs + 1):
    t0 = time.time()
    print(f"epoch {epoch}/{args.epochs} train…", flush=True)
    tr = run_epoch(teacher, student, proj_t, proj_s, train_loader, opt, device, args.temperature, train=True, bank=bank)
    print(f"epoch {epoch}/{args.epochs} val…", flush=True)
    va = run_epoch(teacher, student, proj_t, proj_s, val_loader, None, device, args.temperature, train=False, bank=bank)
    sched.step()
    row = {
        "epoch": epoch,
        "train_loss": tr["loss"],
        "train_acc": tr["acc"],
        "train_acc_k": tr["acc_k"],
        "val_loss": va["loss"],
        "val_acc": va["acc"],
        "val_acc_k": va["acc_k"],
        "lr": opt.param_groups[0]["lr"],
        "sec": round(time.time() - t0, 1),
    }
    history.append(row)
    acc_k = ""
    if args.neg_bank > 0:
      acc_k = f" accK={tr['acc_k']:.3f}/{va['acc_k']:.3f}"
    print(
        f"epoch {epoch:03d} loss={tr['loss']:.4f}/{va['loss']:.4f} "
        f"acc={tr['acc']:.3f}/{va['acc']:.3f}{acc_k} lr={row['lr']:.1e} sec={row['sec']}"
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
        "optimizer": opt.state_dict(),
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
  if args.fail_if_train_loss_above is not None and history:
    last_loss = float(history[-1]["train_loss"])
    if last_loss >= args.fail_if_train_loss_above:
      print(
          f"FAIL train loss {last_loss:.4f} >= {args.fail_if_train_loss_above} "
          "(N=8 chance is 2.079; full job must not start)"
      )
      return 1
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
