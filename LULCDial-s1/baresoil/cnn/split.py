"""Same MD5 70/30 split as build_instruct_s1 / build_bench."""

from __future__ import annotations

import hashlib

DEFAULT_TRAIN_RATIO = 0.7
NUM_CLASSES = 14  # MultiSenGE OCSGE IDs 1–14


def split_bucket(stem: str, train_ratio: float = DEFAULT_TRAIN_RATIO) -> str:
    h = int(hashlib.md5(stem.encode()).hexdigest(), 16) % 1000
    threshold = int(train_ratio * 1000)
    return "train" if h < threshold else "val"


def label_ids_to_multihot(label_ids: list[int], num_classes: int = NUM_CLASSES) -> list[float]:
    vec = [0.0] * num_classes
    for cid in label_ids:
        if 1 <= cid <= num_classes:
            vec[cid - 1] = 1.0
    return vec
