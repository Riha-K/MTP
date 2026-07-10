""" Building  HF shards from MultiSenGE data."""

from __future__ import annotations

# Location/commands:
#   python -m baresoil.build_instruct_s1 ^
#     --labels-dir LULCDial-s1/data/baresoil_s1/ai4lcc/multisenge/labels ^
#     --s1-dir LULCDial-s1/data/baresoil_s1/ai4lcc/multisenge/s1 ^
#     --out-dir LULCDial-s1/data/baresoil_s1/shards/ai4lcc_ge_train ^
#     --split all

import argparse
import hashlib #train/val split; reproducibility
import json
import sys
from pathlib import Path
from datasets import Dataset, Features, Image, Value #huggingface dataset library
from tqdm import tqdm #progress bar

from .instruct_templates import build_classify_qa, build_dialogue_turns
from .patch_meta import iter_patches, pick_available_s1_file, pick_s1_path
from .s1_vh_io import read_s1_vh_db, vh_db_to_pil

#train/val split (90/10)
def _split_bucket(stem: str, train_ratio: float = 0.9) -> str:
    h = int(hashlib.md5(stem.encode()).hexdigest(), 16) % 1000
    threshold = int(train_ratio * 1000)
    return "train" if h < threshold else "val"

# stem = "31UFQ_3341_6939"
#   → md5 → hex string like "c4a1..."
#   → int(hex, 16) → huge number
#   → % 1000 → e.g. 342
#   → 342 < 900 → train

_SHARD_FEATURES = Features(
    {
        "jpg": Image(mode="F"),
        "conversations": Value("string"),
    }
)


class _ShardExampleGen:
    """Stream examples to disk — avoids MemoryError on full MultiSenGE train."""

    def __init__(
        self,
        patches: list,
        s1_dir: Path,
        split: str,
        skip_missing_s1: bool,
    ) -> None:
        self.patches = patches
        self.s1_dir = s1_dir
        self.split = split
        self.skip_missing_s1 = skip_missing_s1
        self.skipped = {"missing_s1": 0, "bad_tif": 0}
        self.patch_count = 0

    def __call__(self):
        self.skipped = {"missing_s1": 0, "bad_tif": 0}
        self.patch_count = 0
        for patch in tqdm(self.patches, desc=f"build-{self.split}"):
            s1_name = pick_available_s1_file(patch, self.s1_dir)
            if not s1_name:
                self.skipped["missing_s1"] += 1
                continue

            s1_path = pick_s1_path(self.s1_dir, s1_name)
            if s1_path is None:
                self.skipped["missing_s1"] += 1
                if self.skip_missing_s1:
                    continue
                raise FileNotFoundError(f"S1 file not found: {s1_name} under {self.s1_dir}")

            try:
                vh = read_s1_vh_db(s1_path)
                pil_img = vh_db_to_pil(vh)
            except Exception:
                self.skipped["bad_tif"] += 1
                continue

            present_names = patch.label_names
            q_cls, a_cls = build_classify_qa(present_names)
            dialogue_conv = build_dialogue_turns(patch.label_ids, present_names)

            yield {
                "jpg": pil_img,
                "conversations": json.dumps(
                    [
                        {"from": "human", "value": q_cls},
                        {"from": "gpt", "value": a_cls},
                    ]
                ),
            }
            yield {
                "jpg": pil_img,
                "conversations": json.dumps(dialogue_conv),
            }
            self.patch_count += 1


#build shard
def build_shard(
    labels_dir: Path,
    s1_dir: Path,
    out_dir: Path,
    split: str, #{train/val only}
    max_patches: int | None,
    skip_missing_s1: bool,
) -> dict:
    patches = iter_patches(labels_dir) #iterate 8157
    if split in ("train", "val"):
        patches = [p for p in patches if _split_bucket(p.stem) == split]
    if max_patches is not None:
        patches = patches[:max_patches]

    gen = _ShardExampleGen(
        patches=patches,
        s1_dir=s1_dir,
        split=split,
        skip_missing_s1=skip_missing_s1,
    )

    # Stream to Arrow cache then save — do not hold all PIL images in RAM
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = Dataset.from_generator(gen, features=_SHARD_FEATURES)
    ds.save_to_disk(str(out_dir))

    manifest = {
        "split": split,
        "num_samples": len(ds),
        "num_patches": gen.patch_count,
        "skipped": gen.skipped,
        "labels_dir": str(labels_dir),
        "s1_dir": str(s1_dir),
        "out_dir": str(out_dir),
        "label_taxonomy": "AI4LCC OCSGE 14-class",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest

#Parse CLI args and run build_shard()
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build AI4LCC S1 instruction shards from MultiSenGE")
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--s1-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True) #write shard
    parser.add_argument("--split", choices=["train", "val", "all"], default="train")
    parser.add_argument("--max-patches", type=int, default=None, help="Cap number of patches (each -> 2 QA)")
    parser.add_argument("--fail-on-missing-s1", action="store_true")
    args = parser.parse_args(argv)

    if not args.labels_dir.is_dir(): #folder check
        print(f"Labels dir not found: {args.labels_dir}", file=sys.stderr)
        return 1
    if not args.s1_dir.is_dir():
        print(f"S1 dir not found: {args.s1_dir}", file=sys.stderr)
        print("Extract MultiSenGE s1.tgz into that folder first.", file=sys.stderr)
        return 1

    #build shards call()
    splits = ["train", "val"] if args.split == "all" else [args.split]
    for sp in splits:
        out = args.out_dir if args.split != "all" else args.out_dir.parent / f"{args.out_dir.name}_{sp}"
        manifest = build_shard(
            labels_dir=args.labels_dir,
            s1_dir=args.s1_dir,
            out_dir=out,
            split=sp,
            max_patches=args.max_patches,
            skip_missing_s1=not args.fail_on_missing_s1,
        )
        print(json.dumps(manifest, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
