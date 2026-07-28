"""Build MultiSenNA bench JSONL for zero-shot transfer evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from ..instruct_templates import build_classify_qa, build_dialogue_turns
from ..patch_meta import iter_patches, pick_available_s1_file, pick_s1_path
from ..s1_vh_io import read_s1_vh_db, vh_db_to_preview_png
from ..taxonomy import format_present_class_names


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--s1-dir", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--preview-dir", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    rows = []
    skipped_missing_s1 = 0

    for patch in tqdm(iter_patches(args.labels_dir), desc="multisenna-bench"):
        if args.max_samples is not None and len(rows) >= args.max_samples:
            break

        s1_name = pick_available_s1_file(patch, args.s1_dir)
        if not s1_name:
            skipped_missing_s1 += 1
            continue

        s1_path = pick_s1_path(args.s1_dir, s1_name)
        if s1_path is None:
            skipped_missing_s1 += 1
            continue

        present_names = patch.label_names
        q_cls, a_cls = build_classify_qa(present_names)
        dialogue = build_dialogue_turns(patch.label_ids, present_names)

        preview_rel = None
        if args.preview_dir:
            args.preview_dir.mkdir(parents=True, exist_ok=True)
            preview_path = args.preview_dir / f"{patch.stem}.png"
            vh_db_to_preview_png(read_s1_vh_db(s1_path), preview_path)
            preview_rel = str(preview_path)

        rows.append(
            {
                "dataset": "MultiSenNA",
                "patch_id": patch.stem,
                "s1_path": str(s1_path),
                "label_ids": patch.label_ids,
                "label_names": present_names,
                "dominant_class_name": patch.dominant_class_name,
                "present_classes": format_present_class_names(present_names),
                "classify_question": q_cls,
                "classify_answer": a_cls,
                "dialogue": dialogue,
                "preview_png": preview_rel,
            }
        )

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    summary = {
        "dataset": "MultiSenNA",
        "num_rows": len(rows),
        "skipped_missing_s1": skipped_missing_s1,
        "labels_dir": str(args.labels_dir),
        "s1_dir": str(args.s1_dir),
        "out_jsonl": str(args.out_jsonl),
    }
    summary_path = args.out_jsonl.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {len(rows)} MultiSenNA bench rows -> {args.out_jsonl}")
    print(f"Wrote summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
