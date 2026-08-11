"""Smoke checks for CNN validation (index counts + optional model forward)."""

from __future__ import annotations

import argparse
from pathlib import Path

from multisenge_seg.dataset import build_patch_index, summarize_splits


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path("LULCDial-s1/data/lulcdial_s1/ai4lcc/multisenge"),
    )
    p.add_argument("--model-smoke", action="store_true")
    p.add_argument("--skip-index", action="store_true")
    p.add_argument(
        "--max-labels",
        type=int,
        default=None,
        help="Quick smoke: only read first N label JSONs (avoids full 8k scan)",
    )
    p.add_argument(
        "--s2-only",
        action="store_true",
        help="Index without requiring S1 (paper S2 gap counts)",
    )
    args = p.parse_args()

    if not args.skip_index:
        print(f"Scanning {args.data_root.resolve()} …")
        records = build_patch_index(
            args.data_root,
            require_s1=not args.s2_only,
            max_labels=args.max_labels,
        )
        summary = summarize_splits(records)
        print("patches passing tile split + 4-month date filter:", len(records))
        print("splits:", summary)
        if args.max_labels:
            print(f"(partial: first {args.max_labels} label files only)")
        if records:
            r = records[0]
            print("example:", r.patch_id, r.split, "s2 months", sorted(r.s2_by_month))

    if args.model_smoke:
        try:
            from multisenge_seg.model import smoke_forward
        except ImportError as e:
            print("model smoke needs torch:", e)
            return 1
        logits, shape = smoke_forward("cpu")
        print("model smoke logits shape:", shape, "mean", float(logits.mean()))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
