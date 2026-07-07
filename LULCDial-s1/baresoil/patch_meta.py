"""AI4LCC / MultiSenGE patch metadata and label JSON helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .taxonomy import (
    ai4lcc_names_from_ids,
    dominant_ocsge_name,
    parse_ai4lcc_label_ids,
)


@dataclass(frozen=True)
class PatchMeta:
    tile: str  # location id on S2 grid
    x: int  # x coordinate of 2.56 km square patch
    y: int
    json_path: Path
    label_ids: list[int]
    label_names: list[str]
    dominant_class_name: str
    s1_files: list[str]
    s2_files: list[str]

    @property
    def stem(self) -> str:
        return f"{self.tile}_{self.x}_{self.y}"

    @property
    def gr_filename(self) -> str:
        return f"{self.tile}_GR_{self.x}_{self.y}.tif"


def parse_label_json(json_path: Path) -> PatchMeta:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    stem = json_path.stem
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected label filename: {json_path.name}")
    tile = parts[0]
    x, y = int(parts[1]), int(parts[2])

    label_ids = parse_ai4lcc_label_ids(data.get("labels", ""))
    s1_files = [s.strip() for s in data.get("corresponding_s1", "").split(";") if s.strip()]
    s2_files = [s.strip() for s in data.get("corresponding_s2", "").split(";") if s.strip()]

    return PatchMeta(
        tile=tile,
        x=x,
        y=y,
        json_path=json_path,
        label_ids=label_ids,
        label_names=ai4lcc_names_from_ids(label_ids),
        dominant_class_name=dominant_ocsge_name(label_ids),
        s1_files=s1_files,
        s2_files=s2_files,
    )


def iter_patches(labels_dir: str | Path) -> list[PatchMeta]:
    labels_dir = Path(labels_dir)
    patches = [parse_label_json(p) for p in sorted(labels_dir.glob("*.json"))]
    return patches


def pick_s1_path(s1_dir: Path, s1_filename: str) -> Path | None:
    """Resolve S1 tif inside extracted MultiSenGE s1 tree."""
    direct = s1_dir / s1_filename
    if direct.exists():
        return direct
    hits = list(s1_dir.rglob(s1_filename))
    return hits[0] if hits else None


def pick_median_s1_file(meta: PatchMeta) -> str | None:
    if not meta.s1_files:
        return None
    return sorted(meta.s1_files)[len(meta.s1_files) // 2]


def pick_available_s1_file(meta: PatchMeta, s1_dir: Path) -> str | None:
    """Median date among S1 files that actually exist on disk."""
    if not meta.s1_files:
        return None
    available = [f for f in sorted(meta.s1_files) if (s1_dir / f).exists()]
    if not available:
        return None
    return available[len(available) // 2]
