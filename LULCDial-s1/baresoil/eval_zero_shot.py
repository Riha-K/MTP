"""Score zero-shot predictions on AI4LCC-S1-Dialogue-Bench."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _split_classes(text: str) -> list[str]:
    if not text:
        return []
    parts = re.split(r"[;,/]| and |\n", text)
    cleaned = [p.strip().lower() for p in parts if p and p.strip()]
    # Preserve order while removing duplicates.
    return list(dict.fromkeys(cleaned))


def _f1_score(pred: set[str], gt: set[str]) -> float:
    if not pred and not gt:
        return 1.0
    if not pred or not gt:
        return 0.0
    tp = len(pred & gt)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gt) if gt else 0.0
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _dialogue_answers(dialogue: list[dict[str, str]]) -> tuple[str, str]:
    gt_turn1 = ""
    gt_turn2 = ""
    for msg in dialogue:
        if msg.get("from") != "gpt":
            continue
        if not gt_turn1:
            gt_turn1 = msg.get("value", "")
        elif not gt_turn2:
            gt_turn2 = msg.get("value", "")
            break
    return gt_turn1, gt_turn2


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _dump_requests(bench_rows: list[dict[str, Any]], out_jsonl: Path) -> None:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in bench_rows:
            dialogue = row.get("dialogue", [])
            turn1_q = dialogue[0]["value"] if len(dialogue) > 0 else ""
            turn2_q = dialogue[2]["value"] if len(dialogue) > 2 else ""
            request_row = {
                "patch_id": row.get("patch_id"),
                "s1_path": row.get("s1_path"),
                "classify_question": row.get("classify_question", ""),
                "dialogue_turn1_question": turn1_q,
                "dialogue_turn2_question": turn2_q,
            }
            f.write(json.dumps(request_row) + "\n")


def evaluate(
    bench_rows: list[dict[str, Any]],
    pred_by_patch: dict[str, dict[str, Any]],
    classify_key: str,
    turn1_key: str,
    turn2_key: str,
) -> dict[str, Any]:
    used = 0
    missing_predictions = 0

    classify_f1_sum = 0.0
    turn1_acc_sum = 0.0
    turn2_acc_sum = 0.0

    for row in bench_rows:
        patch_id = str(row.get("patch_id", ""))
        pred = pred_by_patch.get(patch_id)
        if pred is None:
            missing_predictions += 1
            continue

        used += 1
        gt_cls = set(_split_classes(str(row.get("classify_answer", ""))))
        pd_cls = set(_split_classes(str(pred.get(classify_key, ""))))
        classify_f1_sum += _f1_score(pd_cls, gt_cls)

        gt_turn1, gt_turn2 = _dialogue_answers(row.get("dialogue", []))
        pd_turn1 = set(_split_classes(str(pred.get(turn1_key, ""))))
        pd_turn2 = set(_split_classes(str(pred.get(turn2_key, ""))))
        gt_turn1_set = set(_split_classes(gt_turn1))
        gt_turn2_set = set(_split_classes(gt_turn2))

        turn1_acc_sum += 1.0 if pd_turn1 == gt_turn1_set else 0.0
        turn2_acc_sum += 1.0 if pd_turn2 == gt_turn2_set else 0.0

    if used == 0:
        raise ValueError("No matched predictions found by patch_id.")

    return {
        "num_bench_rows": len(bench_rows),
        "num_scored_rows": used,
        "missing_predictions": missing_predictions,
        "classification": {
            "example_f1": classify_f1_sum / used,
        },
        "dialogue": {
            "turn1_set_match_accuracy": turn1_acc_sum / used,
            "turn2_set_match_accuracy": turn2_acc_sum / used,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate zero-shot predictions on AI4LCC bench.")
    parser.add_argument("--bench-jsonl", type=Path, required=True)
    parser.add_argument("--pred-jsonl", type=Path, default=None)
    parser.add_argument("--out-metrics", type=Path, required=True)
    parser.add_argument("--dump-requests-jsonl", type=Path, default=None)
    parser.add_argument("--classify-key", type=str, default="pred_classify")
    parser.add_argument("--dialogue-turn1-key", type=str, default="pred_dialogue_turn1")
    parser.add_argument("--dialogue-turn2-key", type=str, default="pred_dialogue_turn2")
    args = parser.parse_args()

    bench_rows = _load_jsonl(args.bench_jsonl)

    if args.dump_requests_jsonl:
        _dump_requests(bench_rows, args.dump_requests_jsonl)
        print(f"Wrote eval request rows -> {args.dump_requests_jsonl}")

    if args.pred_jsonl is None:
        print("No --pred-jsonl provided. Request export done; scoring skipped.")
        return 0

    pred_rows = _load_jsonl(args.pred_jsonl)
    pred_by_patch = {str(r.get("patch_id", "")): r for r in pred_rows if r.get("patch_id")}

    metrics = evaluate(
        bench_rows=bench_rows,
        pred_by_patch=pred_by_patch,
        classify_key=args.classify_key,
        turn1_key=args.dialogue_turn1_key,
        turn2_key=args.dialogue_turn2_key,
    )

    args.out_metrics.parent.mkdir(parents=True, exist_ok=True)
    args.out_metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Wrote metrics -> {args.out_metrics}")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
