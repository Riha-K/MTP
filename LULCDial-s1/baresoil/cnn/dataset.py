"""S1 VH multi-label dataset for ResNet baseline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from ..patch_meta import iter_patches, pick_available_s1_file, pick_s1_path
from ..s1_vh_io import read_s1_vh_db
from ..taxonomy import AI4LCC_CLASS_ID_TO_NAME, format_present_class_names
from .split import DEFAULT_TRAIN_RATIO, NUM_CLASSES, label_ids_to_multihot, split_bucket

# ImageNet mean/std — VH is replicated to 3 channels after [-50,10] → [0,1].
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def vh_db_to_tensor(vh_db: np.ndarray) -> torch.Tensor:
    """Clip dB → [0,1], stack to 3ch, ImageNet-normalize. Shape [3,H,W]."""
    vh = np.asarray(vh_db, dtype=np.float32)
    vh = np.clip(vh, -50.0, 10.0)
    x = (vh + 50.0) / 60.0
    x3 = np.stack([x, x, x], axis=0)
    t = torch.from_numpy(x3)
    mean = torch.tensor(_IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    return (t - mean) / std


def build_train_index(
    labels_dir: Path,
    s1_dir: Path,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for meta in iter_patches(labels_dir):
        if split_bucket(meta.stem, train_ratio) != "train":
            continue
        s1_name = pick_available_s1_file(meta, s1_dir)
        if s1_name is None:
            continue
        s1_path = pick_s1_path(s1_dir, s1_name)
        if s1_path is None:
            continue
        rows.append(
            {
                "patch_id": meta.stem,
                "s1_path": str(s1_path),
                "label_ids": list(meta.label_ids),
                "target": label_ids_to_multihot(meta.label_ids),
            }
        )
        if max_samples is not None and len(rows) >= max_samples:
            break
    return rows


def load_bench_index(
    bench_jsonl: Path,
    s1_root: Path | None = None,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    """Load test bench rows; optionally remap s1 paths under s1_root by filename."""
    rows: list[dict[str, Any]] = []
    with bench_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            s1_path = Path(str(obj.get("s1_path", "")))
            if s1_root is not None:
                cand = s1_root / s1_path.name
                if cand.is_file():
                    s1_path = cand
                else:
                    hits = list(s1_root.rglob(s1_path.name))
                    if hits:
                        s1_path = hits[0]
            label_ids = [int(x) for x in obj.get("label_ids", [])]
            rows.append(
                {
                    "patch_id": str(obj.get("patch_id", "")),
                    "s1_path": str(s1_path),
                    "label_ids": label_ids,
                    "target": label_ids_to_multihot(label_ids),
                }
            )
            if max_samples is not None and len(rows) >= max_samples:
                break
    return rows


class S1MultiLabelDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        vh = read_s1_vh_db(row["s1_path"])
        image = vh_db_to_tensor(vh)
        target = torch.tensor(row["target"], dtype=torch.float32)
        return {
            "image": image,
            "target": target,
            "patch_id": row["patch_id"],
            "s1_path": row["s1_path"],
        }


def multihot_to_class_names(probs_or_bits: np.ndarray | list[float], threshold: float = 0.5) -> str:
    names: list[str] = []
    for i, v in enumerate(probs_or_bits):
        if float(v) >= threshold:
            names.append(AI4LCC_CLASS_ID_TO_NAME[i + 1])
    return format_present_class_names(names)


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image": torch.stack([b["image"] for b in batch], dim=0),
        "target": torch.stack([b["target"] for b in batch], dim=0),
        "patch_id": [b["patch_id"] for b in batch],
        "s1_path": [b["s1_path"] for b in batch],
    }
