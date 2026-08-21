# 10-class run (paper Table 7 / 8)

Same tiles, months, S1+S2, ConvLSTM+Inception as 6-class. Label merge: `taxonomy.py` CLASS10_MAP. Reuse `patch_index.json`.

**Report:** [`RESULTS_RS2023_10CLASS.md`](RESULTS_RS2023_10CLASS.md) · JSON [`results/run_c10_v0/test_metrics.json`](results/run_c10_v0/test_metrics.json).

| Metric | Paper | Ours v0 `best.pt` |
|--------|-------|-------------------|
| W-F1 | 0.8851 | **0.8711** |
| W-Recall | 0.8831 | 0.8604 |
| Kappa | 0.7945 | **0.7588** |

Jobs: train **97047** · test **97153**. Classes: 1–5 urban · 6 arable · 7 vineyards+orchards · 8 grassland · 9 forest/semi-natural · 10 water.

```bash
# already done on PARAM
sbatch multisenge_seg/train_c10.sbatch
sbatch multisenge_seg/eval_c10.sbatch   # best.pt → test_metrics.json
```
