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

## PARAM

```bash
cd ~/MTP/earth2
git pull
sbatch multisenge_seg/train_c10.sbatch
```

Startup: full-train stats (~30–90 min) then ~10 min/epoch, EarlyStop ~20–40 epochs, **~6–8 h**. Close SSH.

If `.err` has CUDA out of memory: `scancel` the job, edit sbatch to `--batch-size 4 --accum-steps 4`, resubmit.

When log shows `done. best val weighted F1`:

```bash
sbatch multisenge_seg/eval_c10.sbatch
sbatch multisenge_seg/eval_c10_last.sbatch
```

Eval **both** checkpoints (6-class lesson: last.pt can beat best.pt on urban F1). Confirm logs say `classes=6` is **wrong** — must say **classes=10**.

Copy JSON off PARAM later:

`checkpoints/run_c10_v0/test_metrics.json`  
`checkpoints/run_c10_v0_last_eval/test_metrics.json`
