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

**Transfer learning:** We use the breast **schedule** only, not breast-style **pretrained weights**. U-TAE is trained **from scratch on MultiSenGE** (no PASTIS/ImageNet init) so comparisons stay fair vs A4 and the RS 2023 paper. P4→P5 `--init-ckpt` is same-dataset continuation, not cross-dataset TL. Full rationale: [`TRAINING_AND_TRANSFER.md`](TRAINING_AND_TRANSFER.md).

## Novelty tracks (not P1/P2 — those clash with U-TAE phases)

| Name | Meaning | Files |
|------|---------|--------|
| **Task M** | MA-UTAE dual-stream + gated/concat fuse | `models/ma_utae.py`, `models/fusion.py` |
| **Task H** | Hierarchical agents A1+A2 (UF/rest, Dense/Sparse) | `heads.py` (`--use-a1 --use-a2`) |
| Train | `train_ma.py`, `train_ma*.sbatch`, `eval_ma_c10.sbatch` | |

**Results folders:** stock concat U-TAE → `results/concat_utae/`; MA-UTAE → `results/ma_utae/` (see `results/README.md`).

### Modality ablations (paper-style S1-only / S2-only)

Like MultiSenGE ConvLSTM-S1 / ConvLSTM-S2: stock U-TAE with `--modality s2` (10ch) or `s1` (2ch).

```bash
# 6c — both heads can queue in parallel
sbatch --exclude=ragpu004,ragpu005,ragpu007 multisenge_utae/train_s2_head.sbatch
sbatch --exclude=ragpu004,ragpu005,ragpu007 multisenge_utae/train_s1_head.sbatch
sbatch --exclude=ragpu004,ragpu005,ragpu007 multisenge_utae/train_s2_full.sbatch
sbatch --exclude=ragpu004,ragpu005,ragpu007 multisenge_utae/train_s1_full.sbatch

# 10c modality ablations (optional table rows)
sbatch --exclude=ragpu004,ragpu005,ragpu007 multisenge_utae/train_s2_c10_head.sbatch
sbatch --exclude=ragpu004,ragpu005,ragpu007 multisenge_utae/train_s1_c10_head.sbatch
# after each best.pt:
sbatch --exclude=ragpu004,ragpu005,ragpu007 multisenge_utae/train_s2_c10_full.sbatch
sbatch --exclude=ragpu004,ragpu005,ragpu007 multisenge_utae/train_s1_c10_full.sbatch
```

**Phases on MA-UTAE:** P4 = `--mode head`, P5 = `--mode full --init-ckpt …/best.pt`. **P3 probes:** stock U-TAE only for now.

```bash
# smoke
python -m multisenge_utae.train_ma --index multisenge_seg/artifacts/patch_index.json \
  --num-classes 6 --mode head --fusion gated --epochs 1 \
  --max-train 4 --max-val 2 --out-dir multisenge_utae/checkpoints/ma_c6_smoke

# PARAM 6c gated (done): train_ma.sbatch → train_ma_full.sbatch
# PARAM 10c Task M — gated main (P4 → P5 → test)
sbatch --exclude=ragpu004,ragpu005,ragpu007 multisenge_utae/train_ma_c10_gated_head.sbatch
# after best.pt:
sbatch --exclude=ragpu004,ragpu005,ragpu007 multisenge_utae/train_ma_c10_gated_full.sbatch
CKPT=multisenge_utae/checkpoints/ma_c10_gated_full_v0/best.pt \
  OUT=multisenge_utae/results/ma_utae/ma_c10_gated_full_v0 \
  sbatch --exclude=ragpu004,ragpu005,ragpu007 multisenge_utae/eval_ma_c10.sbatch

# PARAM 10c concat ablation (parallel OK with gated head)
sbatch --exclude=ragpu004,ragpu005,ragpu007 multisenge_utae/train_ma_c10_concat_head.sbatch
sbatch --exclude=ragpu004,ragpu005,ragpu007 multisenge_utae/train_ma_c10_concat_full.sbatch
CKPT=multisenge_utae/checkpoints/ma_c10_concat_full_v0/best.pt \
  OUT=multisenge_utae/results/ma_utae/ma_c10_concat_full_v0 \
  sbatch --exclude=ragpu004,ragpu005,ragpu007 multisenge_utae/eval_ma_c10.sbatch

# Ablation: --fusion concat · Task H: add --use-a1 --use-a2
```

Start **Task M alone** (gated, no agents); add Task H in a second run.

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
cd ~/MTP/earth2

# Preferred on PARAM (avoid down nodes + crowded ragpu004):
sbatch --exclude=ragpu004,ragpu005,ragpu007 multisenge_utae/train.sbatch

sbatch --exclude=ragpu004,ragpu005,ragpu007 multisenge_utae/probe.sbatch      # P3 L0-L3 probes
sbatch multisenge_utae/probe_smoke.sbatch   # P3 smoke (~24 train / 12 val patches)
sbatch multisenge_utae/train_full.sbatch    # P5 full fine-tune after head (6-class)
sbatch multisenge_utae/smoke.sbatch         # short GPU smoke

# 10-class (P4 head → P5 full → test). Do NOT reuse 6-class ckpt.
sbatch --exclude=ragpu004,ragpu005,ragpu007 multisenge_utae/train_c10_head.sbatch
# after head best.pt exists:
sbatch --exclude=ragpu004,ragpu005,ragpu007 multisenge_utae/train_c10_full.sbatch
# test vs A4 10c W-F1 0.8711 / kappa 0.7588:
CKPT=multisenge_utae/checkpoints/run_c10_full_v0/best.pt \
OUT=multisenge_utae/results/concat_utae/run_c10_full_v0 \
  sbatch --exclude=ragpu004,ragpu005,ragpu007 multisenge_utae/eval_c10.sbatch
```

Monitor: `squeue -u rihak_iitp` · log: `tail -f multisenge_utae/artifacts/slurm-<JOBID>.out`

If job fails in ~1 min with `.err`: `CONDA_BACKUP_QT_XCB_GL_INTEGRATION: unbound variable` — fixed in sbatch (`set -eo pipefail`, not `-u` with `module purge`). Run `git pull`.

**GPU node status (PARAM):**

```bash
sinfo -N -p gpu -o "%N %T %C %G"
```

`STATE`: `idle` / `mixed` / `down` / `drained` · `GRES`: `gpu:2` per node · `CPUS(A/I/O/T)`: allocated / idle / other / total.

**Jobs per node (find a free GPU slot — need fewer than 2 running on `gpu:2`):**

```bash
for n in ragpu003 ragpu004 ragpu006 ragpu008; do
  echo "=== $n ==="
  squeue -w $n -t R -o "%.10i %.8u %.10M"
done
```

## P3 — layer probes (L0–L3)

Frozen U-TAE encoder; fit a **linear** pixel classifier on train tiles, score on **val** (31UFP+31UGP). Uses P4 `best.pt` for weights + `norm_stats` (encoder was not updated in P4).

```bash
cd ~/MTP/earth2
git pull

# Optional: quick pipeline check (~15 min)
sbatch multisenge_utae/probe_smoke.sbatch

# Full P3 (all train/val patches, ~512 px/patch; allow up to 12h)
sbatch --exclude=ragpu004,ragpu005,ragpu007 multisenge_utae/probe.sbatch
```

Outputs → `multisenge_utae/results/concat_utae/probe_c6_v0/`:

- `L0_linear_metrics.json` … `L3_linear_metrics.json` (full per-class table each)
- `probe_summary_linear.json` + `.md` (headline W-F1 per level)

Override checkpoint: `CKPT=path/to/best.pt sbatch ... probe.sbatch`

## Training curve plot (sir)

Each train run writes `history.json` (mean **train loss** + val metrics per epoch). Plot:

```bash
python -m multisenge_utae.plot_history \
  multisenge_utae/checkpoints/run_c6_head_v0/history.json \
  --title "U-TAE P4 head-only (6-class)"
```

Output: `history_plot.png` next to `history.json`. P5 `train_full.sbatch` runs this automatically at job end.

## Eval on test tile 31UEQ

```bash
python -m multisenge_utae.train \
  --index multisenge_seg/artifacts/patch_index.json \
  --eval-ckpt multisenge_utae/checkpoints/run_c6_head_v0/best.pt \
  --eval-split test \
  --out-dir multisenge_utae/results/concat_utae/run_c6_head_v0
```

## Model

Vendored from [utae-paps](https://github.com/VSainteuf/utae-paps) (MIT). Encoder widths `[64,64,64,128]`, L-TAE `d_model=256`, `n_head=16`.

## Metrics

Same **`multisenge_seg/metrics.py`** as A4 (one definition for sir-facing tables).  
Formula notes (loss, Softmax, U-TAE/L-TAE, multinomial LR probes, W-F1/Kappa): [`../multisenge_seg/FORMULAS.md`](../multisenge_seg/FORMULAS.md).

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
  --metrics multisenge_utae/results/concat_utae/run_c6_head_v0/test_metrics.json \
  --num-classes 6 \
  --title "U-TAE 6-class test" \
  --probe-summary multisenge_utae/results/concat_utae/probe_c6_v0/probe_summary_linear.json
```

Or use `sbatch multisenge_utae/eval.sbatch` (test eval + auto `.md`).

**Results:** P4 head — [`RESULTS_UTAE_6CLASS_HEAD.md`](RESULTS_UTAE_6CLASS_HEAD.md) · val [`results/concat_utae/run_c6_head_v0/best_metrics.json`](results/concat_utae/run_c6_head_v0/best_metrics.json) · test [`results/concat_utae/run_c6_head_v0/test_metrics.json`](results/concat_utae/run_c6_head_v0/test_metrics.json) · P3 [`results/concat_utae/probe_c6_v0/probe_summary_linear.md`](results/concat_utae/probe_c6_v0/probe_summary_linear.md).
