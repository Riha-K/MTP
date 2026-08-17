"""Segmentation metrics for CNN validation (pixel Weighted F1, etc.)."""

from __future__ import annotations

import numpy as np


def _ratio(num: float, den: float) -> float:
    return 0.0 if den == 0 else num / den


def _f1(tp: float, fp: float, fn: float) -> float:
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else (2 * tp) / denom


def accumulate_confusion(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    ignore_index: int = 255,
) -> np.ndarray:
    """Return (C,C) confusion; rows=gt, cols=pred. Classes are 0..C-1."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    mask = target != ignore_index
    pred = pred[mask].ravel()
    target = target[mask].ravel()
    valid = (target >= 0) & (target < num_classes) & (pred >= 0) & (pred < num_classes)
    pred = pred[valid]
    target = target[valid]
    for t, p in zip(target, pred):
        cm[t, p] += 1
    return cm


def scores_from_cm(cm: np.ndarray) -> dict:
    """Paper Table 5 style: per-class Precision / Recall / F1 + support-weighted averages."""
    c = cm.shape[0]
    support = cm.sum(axis=1).astype(np.float64)
    per_p, per_r, per_f1 = [], [], []
    for i in range(c):
        tp = float(cm[i, i])
        fp = float(cm[:, i].sum() - tp)
        fn = float(cm[i, :].sum() - tp)
        per_p.append(_ratio(tp, tp + fp))
        per_r.append(_ratio(tp, tp + fn))
        per_f1.append(_f1(tp, fp, fn))
    per_p_arr = np.array(per_p, dtype=np.float64)
    per_r_arr = np.array(per_r, dtype=np.float64)
    per_f1_arr = np.array(per_f1, dtype=np.float64)
    total = support.sum()
    if total > 0:
        weighted_p = float((per_p_arr * support).sum() / total)
        weighted_r = float((per_r_arr * support).sum() / total)
        weighted_f1 = float((per_f1_arr * support).sum() / total)
    else:
        weighted_p = weighted_r = weighted_f1 = 0.0
    acc = float(np.trace(cm) / cm.sum()) if cm.sum() > 0 else 0.0
    if cm.sum() == 0:
        kappa = 0.0
    else:
        po = acc
        pe = float((cm.sum(axis=0) * cm.sum(axis=1)).sum() / (cm.sum() ** 2))
        kappa = 0.0 if pe == 1.0 else float((po - pe) / (1.0 - pe))
    return {
        "accuracy": acc,
        "weighted_precision": weighted_p,
        "weighted_recall": weighted_r,
        "weighted_f1": weighted_f1,
        "mean_f1": float(per_f1_arr.mean()) if c else 0.0,
        "per_class_precision": per_p_arr.tolist(),
        "per_class_recall": per_r_arr.tolist(),
        "per_class_f1": per_f1_arr.tolist(),
        "support": support.tolist(),
        "kappa": kappa,
    }


def format_prf_table(scores: dict) -> str:
    """Printable Table-5 block: class | Precision | Recall | F1."""
    p = scores["per_class_precision"]
    r = scores["per_class_recall"]
    f = scores["per_class_f1"]
    lines = ["class  Precision  Recall     F1"]
    for i in range(len(f)):
        lines.append(f"{i + 1:5d}  {p[i]:9.4f}  {r[i]:6.4f}  {f[i]:6.4f}")
    lines.append(
        f"W-Avg  {scores['weighted_precision']:9.4f}  "
        f"{scores['weighted_recall']:6.4f}  {scores['weighted_f1']:6.4f}"
    )
    return "\n".join(lines)


def class_weights_from_counts(counts: np.ndarray, ignore_zeros: bool = True) -> np.ndarray:
    """Inverse-frequency weights for Weighted CE (paper-style)."""
    counts = counts.astype(np.float64)
    if ignore_zeros:
        counts = np.maximum(counts, 1.0)
    inv = 1.0 / counts
    return inv / inv.sum() * len(counts)
