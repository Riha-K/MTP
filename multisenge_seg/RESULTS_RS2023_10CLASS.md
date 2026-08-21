# RS 2023 replicate — 10-class test results vs paper

Independent reimplementation of **ConvLSTM+Inception-S1S2** (Wenger et al., *Remote Sensing* 2023, §4.2, Table 7 & 8).

**Report:** v0 **`best.pt`** (val W-F1 peak, epochs 24–25). JSON: [`results/run_c10_v0/test_metrics.json`](results/run_c10_v0/test_metrics.json).

| Item | Value |
|------|--------|
| Split | Test tile **31UEQ** (~610 patches), 10-class |
| Train | PARAM job **97047** |
| Eval | job **97153** · `classes=10` |
| Report ckpt | `checkpoints/run_c10_v0/best.pt` |

---

## Headline vs paper

| Metric | Paper | Ours (v0 best.pt) |
|--------|-------|-------------------|
| W-Avg F1 | **0.8851** | **0.8711** |
| W-Avg Recall | 0.8831 | 0.8604 |
| Kappa | **0.7945** | **0.7588** |

Gap: grassland (low recall) and water (low precision); rare urban class 4 still weak. Mean F1 **0.585** vs W-F1 **0.871** (arable dominates).

---

## Status

10-class **frozen** on v0. Extra reweight/seed runs did not beat this test table — discarded from repo. Next optional: **A5**.
