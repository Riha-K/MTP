"""Layer-wise probes L0-L3 (breast-paper P3): frozen encoder + RF or linear pixel classifier.

Example:
  python -m multisenge_utae.probe_layers \\
    --index multisenge_seg/artifacts/patch_index.json \\
    --num-classes 6 --probe linear \\
    --max-train-patches 32 --max-val-patches 16 \\
    --pixels-per-patch 256 \\
    --out-dir multisenge_utae/results/probe_c6_v0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from multisenge_seg.build_index import records_from_json
from multisenge_seg.dataset import MultiSenGETemporalDataset, PatchRecord, build_patch_index
from multisenge_seg.metrics import format_prf_table, scores_from_cm
from multisenge_seg.taxonomy import num_output_classes
from multisenge_seg.train import estimate_channel_stats, set_seed
from multisenge_utae.data import batch_positions, collate_utae
from multisenge_utae.models import UTAE

LEVELS = ("L0", "L1", "L2", "L3")


def _pool_level(feat: torch.Tensor, level: str) -> torch.Tensor:
    """Return B x D x H x W feature map for a probe level."""
    if level == "L3":
        return feat
    # L0-L2: B,T,C,H,W -> mean over time
    return feat.mean(dim=1)


@torch.no_grad()
def collect_probe_samples(
    model: UTAE,
    loader: DataLoader,
    device: torch.device,
    level: str,
    pixels_per_patch: int,
    num_classes: int,
    max_patches: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    n_patches = 0
    rng = np.random.default_rng(0)
    for batch in loader:
        x = batch["x"].to(device)
        mask = batch["mask"].numpy()
        bp = batch_positions(x.shape[0], device)
        levels = model.encode_levels(x, batch_positions=bp)
        feat = _pool_level(levels[level], level)
        feat_np = feat.cpu().numpy()
        b, c, h, w = feat_np.shape
        for i in range(b):
            m = mask[i]
            valid = m != 255
            if not valid.any():
                continue
            yy, xx = np.where(valid)
            lab = m[yy, xx]
            n = min(pixels_per_patch, len(yy))
            if n < len(yy):
                pick = rng.choice(len(yy), size=n, replace=False)
                yy, xx, lab = yy[pick], xx[pick], lab[pick]
            # map label coords to feature map resolution
            fy = np.clip((yy * h) // m.shape[0], 0, h - 1)
            fx = np.clip((xx * w) // m.shape[1], 0, w - 1)
            feats = feat_np[i, :, fy, fx].T
            xs.append(feats.astype(np.float32))
            ys.append(lab.astype(np.int64))
            n_patches += 1
            if max_patches is not None and n_patches >= max_patches:
                break
        if max_patches is not None and n_patches >= max_patches:
            break
    if not xs:
        raise RuntimeError(f"no probe pixels collected for level={level}")
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def fit_probe(x_train: np.ndarray, y_train: np.ndarray, probe: str, num_classes: int):
    if probe == "rf":
        from sklearn.ensemble import RandomForestClassifier

        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            n_jobs=-1,
            class_weight="balanced_subsample",
            random_state=0,
        )
        clf.fit(x_train, y_train)
        return clf
    if probe == "linear":
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            multi_class="multinomial",
            n_jobs=-1,
            random_state=0,
        )
        clf.fit(x_train, y_train)
        return clf
    raise ValueError(f"unknown probe {probe}")


def predict_probe(clf, x: np.ndarray) -> np.ndarray:
    return clf.predict(x).astype(np.int64)


def scores_from_preds(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    valid = (y_true >= 0) & (y_true < num_classes) & (y_pred >= 0) & (y_pred < num_classes)
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return scores_from_cm(cm)


def load_model(args, input_dim: int, n_cls: int, device: torch.device) -> UTAE:
    model = UTAE(input_dim=input_dim, num_classes=n_cls).to(device)
    if args.ckpt:
        try:
            ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(args.ckpt, map_location=device)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state, strict=False)
        print("loaded ckpt", args.ckpt)
    for p in model.parameters():
        p.requires_grad = False
    return model


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path("LULCDial-s1/data/lulcdial_s1/ai4lcc/multisenge"))
    p.add_argument("--index", type=Path, default=None)
    p.add_argument("--num-classes", type=int, default=6, choices=[6, 10])
    p.add_argument("--probe", type=str, default="linear", choices=["linear", "rf"])
    p.add_argument("--ckpt", type=Path, default=None, help="Optional encoder weights (else random init)")
    p.add_argument("--pixels-per-patch", type=int, default=512)
    p.add_argument("--max-train-patches", type=int, default=None)
    p.add_argument("--max-val-patches", type=int, default=None)
    p.add_argument("--stats-patches", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=Path("multisenge_utae/results/probe_c6_v0"))
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    set_seed(args.seed)
    n_cls = num_output_classes(args.num_classes)
    if args.index and args.index.is_file():
        records = records_from_json(args.index)
    else:
        records = build_patch_index(args.data_root)

    stats = estimate_channel_stats(records, args.num_classes, max_patches=args.stats_patches)
    train_ds = MultiSenGETemporalDataset(
        records,
        "train",
        num_classes=args.num_classes,
        augment=False,
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
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_utae,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_utae,
    )
    sample = collate_utae([train_ds[0]])
    input_dim = int(sample["x"].shape[2])
    device = torch.device(args.device)
    model = load_model(args, input_dim, n_cls, device)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}
    for level in LEVELS:
        print(f"=== probe level {level} ({args.probe}) ===")
        x_tr, y_tr = collect_probe_samples(
            model,
            train_loader,
            device,
            level,
            args.pixels_per_patch,
            n_cls,
            args.max_train_patches,
        )
        x_va, y_va = collect_probe_samples(
            model,
            val_loader,
            device,
            level,
            args.pixels_per_patch,
            n_cls,
            args.max_val_patches,
        )
        print(f"train pixels={len(y_tr)} val pixels={len(y_va)} dim={x_tr.shape[1]}")
        clf = fit_probe(x_tr, y_tr, args.probe, n_cls)
        pred = predict_probe(clf, x_va)
        scores = scores_from_preds(y_va, pred, n_cls)
        summary[level] = scores
        print(
            f"{level} val wF1={scores['weighted_f1']:.4f} "
            f"wSens={scores['weighted_sensitivity']:.4f} "
            f"wSpec={scores['weighted_specificity']:.4f} "
            f"acc={scores['accuracy']:.4f} kappa={scores['kappa']:.4f}"
        )
        print(format_prf_table(scores))
        (args.out_dir / f"{level}_{args.probe}_metrics.json").write_text(
            json.dumps(scores, indent=2),
            encoding="utf-8",
        )

    (args.out_dir / f"probe_summary_{args.probe}.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print("wrote", args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
