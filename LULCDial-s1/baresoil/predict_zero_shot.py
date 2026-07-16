"""Run EarthDial_4B_MS zero-shot inference on AI4LCC / MultiSenNA bench JSONL.

Writes prediction rows for ``eval_zero_shot.py``:
  patch_id, pred_classify, pred_dialogue_turn1, pred_dialogue_turn2

Example (PARAM GPU node, from LULCDial-s1):

  export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
  python -m baresoil.predict_zero_shot \\
    --bench-jsonl data/baresoil_s1/bench/v0.1/ai4lcc_val.jsonl \\
    --s1-root data/baresoil_s1/ai4lcc/multisenge/s1_val_bench \\
    --checkpoint /home/rihak_iitp/EarthDial_Models/EarthDial_4B_MS \\
    --out-pred-jsonl data/baresoil_s1/bench/v0.1/ai4lcc_val_predictions.jsonl \\
    --max-samples 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .s1_vh_io import read_s1_vh_db, vh_db_to_pil


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _resolve_s1_path(s1_path: str, data_root: Path, s1_root: Path | None) -> Path:
    raw = Path(str(s1_path).replace("\\", "/"))
    if s1_root is not None:
        return s1_root / raw.name
    if raw.is_file():
        return raw
    cand = data_root / raw
    if cand.is_file():
        return cand
    return cand


def _dialogue_questions(row: dict[str, Any]) -> tuple[str, str]:
    dialogue = row.get("dialogue") or []
    if len(dialogue) >= 3:
        return str(dialogue[0].get("value", "")), str(dialogue[2].get("value", ""))
    # requests JSONL shape
    return (
        str(row.get("dialogue_turn1_question", "")),
        str(row.get("dialogue_turn2_question", "")),
    )


def _prepare_pixel_values(model, s1_path: Path, image_size: int):
    import torch
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    # Match earthdial.train.dataset.build_transform(..., normalize_type='s1') eval path
    # without importing dataset.py (avoids cv2/imageio side deps on PARAM).
    s1_mean = (-20.26,)
    s1_std = (5.91,)
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Lambda(lambda x: x.unsqueeze(0) if x.ndim == 2 else x),
            T.Normalize(mean=s1_mean, std=s1_std),
            T.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        ]
    )

    vh = read_s1_vh_db(s1_path)
    pil = vh_db_to_pil(vh)
    pixel_values = transform(pil).unsqueeze(0)
    pixel_values = model.sequential_vit_features(pixel_values, "bilinear")
    return pixel_values.to(dtype=torch.bfloat16).cuda()


def _chat(model, tokenizer, pixel_values, question: str, generation_config: dict, history=None):
    return model.chat(
        tokenizer=tokenizer,
        pixel_values=pixel_values,
        question=question,
        generation_config=generation_config,
        history=history,
        return_history=True,
        verbose=False,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EarthDial zero-shot bench inference (S1 VH).")
    parser.add_argument("--bench-jsonl", type=Path, required=True)
    parser.add_argument("--out-pred-jsonl", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/home/rihak_iitp/EarthDial_Models/EarthDial_4B_MS",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("."),
        help="Repo-relative root for resolving s1_path (usually LULCDial-s1).",
    )
    parser.add_argument(
        "--s1-root",
        type=Path,
        default=None,
        help="If set, load TIFF by filename under this folder (ignores path prefix in JSONL).",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip patch_ids already present in --out-pred-jsonl.",
    )
    args = parser.parse_args(argv)

    # EarthDial lives under src/
    src_dir = Path(__file__).resolve().parents[1] / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    import torch
    from earthdial.model.internvl_chat import InternVLChatModel
    from transformers import AutoTokenizer

    rows = _load_jsonl(args.bench_jsonl)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    done: set[str] = set()
    if args.resume and args.out_pred_jsonl.is_file():
        for prev in _load_jsonl(args.out_pred_jsonl):
            pid = str(prev.get("patch_id", ""))
            if pid:
                done.add(pid)
        print(f"Resume: skipping {len(done)} existing predictions")

    print(f"Loading checkpoint: {args.checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True, use_fast=False)
    model = InternVLChatModel.from_pretrained(
        args.checkpoint,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
    ).eval().cuda()
    image_size = model.config.force_image_size or model.config.vision_config.image_size

    generation_config = dict(
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.temperature > 0,
        temperature=args.temperature if args.temperature > 0 else 0.0,
    )

    args.out_pred_jsonl.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume and args.out_pred_jsonl.is_file() else "w"
    written = 0
    skipped_missing = 0
    skipped_bad_tif = 0

    with args.out_pred_jsonl.open(mode, encoding="utf-8") as out_f:
        for row in tqdm(rows, desc="zs-infer"):
            patch_id = str(row.get("patch_id", ""))
            if not patch_id or patch_id in done:
                continue

            s1_path = _resolve_s1_path(str(row.get("s1_path", "")), args.data_root, args.s1_root)
            if not s1_path.is_file():
                skipped_missing += 1
                print(f"missing S1: {s1_path}", file=sys.stderr)
                continue

            classify_q = str(row.get("classify_question", ""))
            turn1_q, turn2_q = _dialogue_questions(row)

            try:
                with torch.no_grad():
                    pixel_values = _prepare_pixel_values(model, s1_path, image_size)
                    pred_classify, _ = _chat(model, tokenizer, pixel_values, classify_q, generation_config)
                    pred_turn1, history = _chat(model, tokenizer, pixel_values, turn1_q, generation_config)
                    pred_turn2, _ = _chat(
                        model, tokenizer, pixel_values, turn2_q, generation_config, history=history
                    )
            except Exception as exc:  # noqa: BLE001 — keep long jobs alive on corrupt uploads
                skipped_bad_tif += 1
                print(f"bad S1 (skip): {s1_path} ({type(exc).__name__}: {exc})", file=sys.stderr)
                continue

            out_f.write(
                json.dumps(
                    {
                        "patch_id": patch_id,
                        "pred_classify": pred_classify,
                        "pred_dialogue_turn1": pred_turn1,
                        "pred_dialogue_turn2": pred_turn2,
                        "s1_path": str(s1_path),
                    }
                )
                + "\n"
            )
            out_f.flush()
            written += 1

    print(
        json.dumps(
            {
                "wrote": written,
                "skipped_missing_s1": skipped_missing,
                "skipped_bad_tif": skipped_bad_tif,
                "out_pred_jsonl": str(args.out_pred_jsonl),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
