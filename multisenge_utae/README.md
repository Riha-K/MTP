# MultiSenGE U-TAE (A5)

U-TAE temporal segmentation on MultiSenGE, following the breast-cancer transfer-learning workflow (layer probes, head-only train, full fine-tune) while keeping the **geographic tile split** from `multisenge_seg`.

## Class count

**Start with 6 classes** (`--num-classes 6`, Wenger RS 2023 Table 5). Run 10-class (`--num-classes 10`, Table 6) as a second pass after 6-class is stable.

## Data split (tile method)

Same as `multisenge_seg/PROTOCOL.md`:

| Split | Tiles | ~Patches |
|-------|-------|----------|
| train | 8 tiles (32UMV, 32ULU, 32TLT, 31UGQ, 31TFN, 31UFQ, 31UFR, 32ULV) | ~3369 |
| val | 31UFP, 31UGP | ~1911 |
| test | 31UEQ | ~610 |

Reuse `multisenge_seg/artifacts/patch_index.json` (build with `python -m multisenge_seg.build_index`).

## Input tensor

Per date: concat **10 S2 + 2 S1 (VV, VH)** channels -> `B x 4 x 12 x 256 x 256`.

## Breast-paper phases on U-TAE

| Phase | Script | What |
|-------|--------|------|
| P3 | `probe_layers.py` | Frozen encoder; L0-L3 pixel probes (linear or RF) |
| P4 | `train.py --mode head` | Freeze encoder + L-TAE; train decoder/head |
| P5 | `train.py --mode full` | Fine-tune all weights (`--init-ckpt` from best head run) |

## Quick start (smoke)

```bash
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"

python -m multisenge_utae.train \
  --index multisenge_seg/artifacts/patch_index.json \
  --num-classes 6 --mode head --epochs 2 \
  --max-train 8 --max-val 4 \
  --out-dir multisenge_utae/checkpoints/run_c6_head_smoke
```

## PARAM (full run)

```bash
sbatch multisenge_utae/train.sbatch      # head, 6-class
sbatch multisenge_utae/train_full.sbatch # full fine-tune after head
sbatch multisenge_utae/probe.sbatch      # L0-L3 probes
```

## Eval on test tile 31UEQ

```bash
python -m multisenge_utae.train \
  --index multisenge_seg/artifacts/patch_index.json \
  --eval-ckpt multisenge_utae/checkpoints/run_c6_head_v0/best.pt \
  --eval-split test \
  --out-dir multisenge_utae/results/run_c6_head_v0
```

## Model

Vendored from [utae-paps](https://github.com/VSainteuf/utae-paps) (MIT). Encoder widths `[64,64,64,128]`, L-TAE `d_model=256`, `n_head=16`.

## Metrics

Same **`multisenge_seg/metrics.py`** as A4 (one definition for sir-facing tables):

| Metric | JSON key | Notes |
|--------|----------|-------|
| Per-class Precision | `per_class_precision` | Table 5 |
| Per-class Recall | `per_class_recall` | Table 5 |
| Per-class Sensitivity | `per_class_sensitivity` | = Recall (breast-paper name) |
| Per-class Specificity | `per_class_specificity` | One-vs-rest |
| Per-class F1 | `per_class_f1` | Table 5 |
| Weighted F1 | `weighted_f1` | **Headline** vs A4 |
| Weighted Precision / Recall | `weighted_precision`, `weighted_recall` | W-Avg row |
| Weighted Sensitivity / Specificity | `weighted_sensitivity`, `weighted_specificity` | Breast-TL style W-Avg |
| Kappa | `kappa` | Table 6 |
| Accuracy | `accuracy` | Logged |
| Confusion matrix | `confusion_matrix` | Rows=GT, cols=pred |

Copy JSON from PARAM into `multisenge_utae/results/` then export markdown for `log.md`:

```bash
python -m multisenge_utae.export_notes \
  --metrics multisenge_utae/results/run_c6_head_v0/test_metrics.json \
  --num-classes 6 \
  --title "U-TAE 6-class test" \
  --probe-summary multisenge_utae/results/probe_c6_v0/probe_summary_linear.json
```

Or use `sbatch multisenge_utae/eval.sbatch` (test eval + auto `.md`).
