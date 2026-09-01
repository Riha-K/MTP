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
from multisenge_utae.export_notes import probe_summary_markdown
from multisenge_utae.models import UTAE

LEVELS = ("L0", "L1", "L2", "L3")


def load_norm_stats(
    ckpt_path: Path | None,
    records: list[PatchRecord],
    num_classes: int,
    stats_patches: int,
) -> dict:
    if ckpt_path and ckpt_path.is_file():
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location="cpu")
        stats = ckpt.get("norm_stats") if isinstance(ckpt, dict) else None
        if not stats:
            stats_path = ckpt_path.parent / "norm_stats.json"
            if stats_path.is_file():
                stats = json.loads(stats_path.read_text(encoding="utf-8"))
        if stats:
            print("using norm_stats from", ckpt_path)
            return stats
    return estimate_channel_stats(records, num_classes, max_patches=stats_patches)


def _pool_level(feat: torch.Tensor, level: str) -> torch.Tensor:
    """Return B x C x H x W feature map for a probe level."""
    if feat.dim() == 5:
        return feat.mean(dim=1)
    if feat.dim() == 4:
        return feat
    raise ValueError(f"level {level}: expected 4D or 5D features, got {feat.dim()}D")


def _sample_pixel_features(
    feat_bchw: np.ndarray,
    batch_i: int,
    yy: np.ndarray,
    xx: np.ndarray,
    mask_shape: tuple[int, int],
) -> np.ndarray:
    """Map mask pixels to feature vectors; shape (n_pixels, n_channels)."""
    plane = feat_bchw[batch_i]
    c, fh, fw = plane.shape
    mh, mw = mask_shape
    n = len(yy)
    fy = np.clip((yy.astype(np.int64) * fh) // mh, 0, fh - 1)
    fx = np.clip((xx.astype(np.int64) * fw) // mw, 0, fw - 1)
    if fy.shape != (n,) or fx.shape != (n,):
        raise RuntimeError(f"fy/fx length mismatch: {fy.shape}, {fx.shape}, n={n}")
    # Pairwise indices (C, N) — avoid broadcast when fy/fx lengths differ.
    sampled = plane[:, fy, fx]
    if sampled.shape != (c, n):
        raise RuntimeError(
            f"feature/label count mismatch: got {sampled.shape}, expected ({c}, {n})"
        )
    return sampled.T.astype(np.float32)


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
        if feat.dim() != 4:
            raise RuntimeError(f"level {level}: expected BCHW after pool, got {tuple(feat.shape)}")
        feat_np = feat.cpu().numpy()
        b = feat_np.shape[0]
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
            feats = _sample_pixel_features(feat_np, i, yy, xx, m.shape)
            if feats.shape[0] != len(lab):
                raise RuntimeError(
                    f"level {level} patch {n_patches}: feats {feats.shape[0]} != labels {len(lab)}"
                )
            xs.append(feats)
            ys.append(lab.astype(np.int64))
            n_patches += 1
            if max_patches is not None and n_patches >= max_patches:
                break
        if max_patches is not None and n_patches >= max_patches:
            break
    if not xs:
        raise RuntimeError(f"no probe pixels collected for level={level}")
    x_all = np.concatenate(xs, axis=0)
    y_all = np.concatenate(ys, axis=0)
    if x_all.shape[0] != y_all.shape[0]:
        raise RuntimeError(
            f"level {level}: x/y length mismatch {x_all.shape[0]} vs {y_all.shape[0]}"
        )
    return x_all, y_all


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

        solver = "saga" if len(y_train) > 100_000 else "lbfgs"
        clf = LogisticRegression(
            max_iter=1000 if solver == "saga" else 2000,
            solver=solver,
            tol=1e-3,
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


def load_model(ckpt_path: Path | None, input_dim: int, n_cls: int, device: torch.device) -> UTAE:
    if ckpt_path and ckpt_path.is_file():
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location=device)
        if not isinstance(ckpt, dict) or "model" not in ckpt:
            raise RuntimeError(f"expected train checkpoint with 'model' key: {ckpt_path}")
        n_cls = int(ckpt.get("num_classes", n_cls))
        input_dim = int(ckpt.get("input_dim", input_dim))
        model = UTAE(input_dim=input_dim, num_classes=n_cls).to(device)
        model.load_state_dict(ckpt["model"])
        print("loaded ckpt", ckpt_path, f"(encoder frozen in P4; probes use L0-L3 features)")
    else:
        model = UTAE(input_dim=input_dim, num_classes=n_cls).to(device)
        print("random-init encoder (no --ckpt)")
    for p in model.parameters():
        p.requires_grad = False
    return model


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path("LULCDial-s1/data/lulcdial_s1/ai4lcc/multisenge"))
    p.add_argument("--index", type=Path, default=None)
    p.add_argument("--num-classes", type=int, default=6, choices=[6, 10])
    p.add_argument("--probe", type=str, default="linear", choices=["linear", "rf"])
    p.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="P4 head checkpoint (uses same norm_stats as train/eval)",
    )
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

    stats = load_norm_stats(args.ckpt, records, args.num_classes, args.stats_patches)
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
    model = load_model(args.ckpt, input_dim, n_cls, device)

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

    summary_path = args.out_dir / f"probe_summary_{args.probe}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md_path = args.out_dir / f"probe_summary_{args.probe}.md"
    md_path.write_text(probe_summary_markdown(summary, args.probe), encoding="utf-8")
    print("wrote", args.out_dir, summary_path.name, md_path.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
