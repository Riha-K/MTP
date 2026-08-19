# multisenge_seg — ConvLSTM+Inception-S1S2 replication

Replicate **Wenger et al. Remote Sensing 2023** on MultiSenGE, then advance under the same protocol.

| Doc | Purpose |
|-----|---------|
| [`PROTOCOL.md`](PROTOCOL.md) | Frozen setting (dates, tiles, classes) |
| [`RESULTS_RS2023_6CLASS.md`](RESULTS_RS2023_6CLASS.md) | Test P/R/F1 vs paper Table 5/6 |
| [`TABLE5_TEST_FOR_SIR.md`](TABLE5_TEST_FOR_SIR.md) | Paper-format tables · best vs last · epoch numbers |
| [`results/run_c6_v0/last_test_metrics.json`](results/run_c6_v0/last_test_metrics.json) | **Report** — v0 last.pt epoch 25 |
| [`results/run_c6_v0/test_metrics.json`](results/run_c6_v0/test_metrics.json) | v0 best.pt epoch 5 |
| `dataset.py` | Index + Dataset (10-band S2, VV/VH S1) |
| `model.py` | ConvLSTM + Naive Inception + **VGG-16 U-Net** |
| `train.py` / `train.sbatch` | Weighted CE train loop (PARAM) |
| `train_c10.sbatch` / `eval_c10.sbatch` | **10-class** Table 7/8 ([`CLASS10.md`](CLASS10.md)) |
| `taxonomy.py` / `splits.py` | 6/10-class maps, geo tiles |

## Data

`LULCDial-s1/data/lulcdial_s1/ai4lcc/multisenge/{labels,s1,s2,ground_reference}`

## Smoke (laptop)

```bash
python -m multisenge_seg.smoke_index
python -m multisenge_seg.smoke_index --model-smoke   # needs torch
```

## Train (PARAM)

See **[`PARAM_TRANSFER.md`](PARAM_TRANSFER.md)** for copy steps (git pull + scp/rsync S2/GR; reuse S1 if already there).

```bash
export PYTHONPATH=$PWD
python -m multisenge_seg.build_index
python -m multisenge_seg.train --index multisenge_seg/artifacts/patch_index.json \
  --num-classes 6 --epochs 80 --batch-size 2 --out-dir multisenge_seg/checkpoints/run_c6_v0
# or: sbatch multisenge_seg/train.sbatch
```

Held-out **test** (tile `31UEQ`, paper Table 5):

```bash
sbatch multisenge_seg/eval.sbatch
# or:
python -m multisenge_seg.train \
  --index multisenge_seg/artifacts/patch_index.json \
  --eval-ckpt multisenge_seg/checkpoints/run_c6_v0/best.pt \
  --eval-split test --out-dir multisenge_seg/checkpoints/run_c6_v0
```

Smoke: `--max-train 8 --max-val 4 --epochs 1`

## Remaining gaps vs paper

- Exact MultiSenGE-Tools `has_days_gap_s2` combinatorics (we use greedy 17-day; counts matched after adding tile **32ULV**)
- Full train-set channel stats (default subset 64 patches; raise `--stats-patches` on PARAM)
