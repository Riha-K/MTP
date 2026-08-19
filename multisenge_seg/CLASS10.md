# 10-class run (paper Table 7 / 8)

Same tiles, months, S1+S2, ConvLSTM+Inception as 6-class. Only the **label merge** changes (`taxonomy.py` CLASS10_MAP).

Reuse existing `patch_index.json` — do not rebuild.

## Paper targets (ConvLSTM+Inception-S1S2, test 31UEQ)

| Metric | Paper |
|--------|-------|
| W-Avg F1 | 0.8851 |
| W-Avg Recall (acc-like) | 0.8831 |
| Kappa | 0.7945 |

10 classes: 1–5 urban (same) · 6 arable · 7 vineyards+orchards · 8 grassland · 9 forest/semi-natural · 10 water.

## v0 — frozen replicate (report this vs paper)

Job **97047** · `checkpoints/run_c10_v0/best.pt` · eval **97153** · **no seed**.

| Metric | Paper | v0 best.pt |
|--------|-------|------------|
| W-F1 | 0.8851 | **0.8711** |
| W-Recall | 0.8831 | 0.8604 |
| Kappa | 0.7945 | **0.7588** |

`last.pt` (eval 97154) W-F1 0.8701 / kappa 0.7566 — do not use. Holes: grassland recall 0.42, water precision 0.30.

## v1 — improvement attempt (not the replicate row)

`--seed 42` · EarlyStop/LR on **kappa** · CE boost **8:2.5** (grass), **4:1.5** (rare urban), **10:0.6** (water was over-predicted, so do not boost it).

```bash
cd ~/MTP/earth2
git pull
sbatch multisenge_seg/train_c10_v1.sbatch
```

Writes `checkpoints/run_c10_v1/` — does **not** overwrite v0.

When log shows `done.`:

```bash
sbatch multisenge_seg/eval_c10_v1.sbatch
sbatch multisenge_seg/eval_c10_v1_last.sbatch
```

Logs must say `classes=10`. Compare test json to v0 **0.8711 / 0.7588**. Keep v0 as the paper-replicate row even if v1 is better.
