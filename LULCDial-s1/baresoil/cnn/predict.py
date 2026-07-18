"""Predict ResNet multi-label on AI4LCC bench → JSONL for eval_zero_shot.

Example:
  python -m baresoil.cnn.predict \\
    --checkpoint checkpoints/resnet18_s1_v0.2/best.pt \\
    --bench-jsonl data/baresoil_s1/bench/v0.2/ai4lcc_test.jsonl \\
    --s1-root data/baresoil_s1/ai4lcc/multisenge/s1_test_bench_v0.2 \\
    --out-pred-jsonl data/baresoil_s1/bench/v0.2/preds/resnet18/ai4lcc_test_predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import S1MultiLabelDataset, collate_batch, load_bench_index, multihot_to_class_names
from .model import load_checkpoint
from .split import NUM_CLASSES


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Predict ResNet multi-label on S1 bench.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bench-jsonl", type=Path, required=True)
    parser.add_argument("--s1-root", type=Path, default=None, help="Optional pack root (remap by filename).")
    parser.add_argument("--out-pred-jsonl", type=Path, required=True)
    parser.add_argument("--arch", type=str, default="resnet18", choices=["resnet18", "resnet50"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args(argv)

    if not args.checkpoint.is_file():
        print(f"Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 1
    if not args.bench_jsonl.is_file():
        print(f"Bench not found: {args.bench_jsonl}", file=sys.stderr)
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(str(args.checkpoint), device=device, arch=args.arch, num_classes=NUM_CLASSES)

    rows = load_bench_index(
        bench_jsonl=args.bench_jsonl,
        s1_root=args.s1_root,
        max_samples=args.max_samples,
    )
    print(f"bench rows={len(rows)} device={device}")

    ds = S1MultiLabelDataset(rows)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_batch,
    )

    args.out_pred_jsonl.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0

    with args.out_pred_jsonl.open("w", encoding="utf-8") as out_f:
        for batch in tqdm(loader, desc="resnet-predict"):
            images = batch["image"].to(device, non_blocking=True)
            try:
                with torch.no_grad():
                    logits = model(images)
                    probs = torch.sigmoid(logits).cpu().numpy()
            except Exception as exc:  # noqa: BLE001
                skipped += len(batch["patch_id"])
                print(f"batch skip ({type(exc).__name__}: {exc})", file=sys.stderr)
                continue

            for i, patch_id in enumerate(batch["patch_id"]):
                pred_classify = multihot_to_class_names(probs[i], threshold=args.threshold)
                out_f.write(
                    json.dumps(
                        {
                            "patch_id": patch_id,
                            "s1_path": batch["s1_path"][i],
                            "pred_classify": pred_classify,
                            "pred_dialogue_turn1": "",
                            "pred_dialogue_turn2": "",
                        }
                    )
                    + "\n"
                )
                written += 1

    print(
        json.dumps(
            {
                "wrote": written,
                "skipped": skipped,
                "out_pred_jsonl": str(args.out_pred_jsonl),
                "threshold": args.threshold,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
