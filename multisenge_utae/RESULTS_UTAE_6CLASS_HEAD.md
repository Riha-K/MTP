# U-TAE A5 — 6-class test results (P4 head-only) vs A4

**Model:** U-TAE · **Phase:** P4 (`--mode head`, frozen encoder + L-TAE)  
**Protocol:** Same geographic split as `multisenge_seg` (test tile **31UEQ**, ~610 patches, 6-class).

| Item | Value |
|------|--------|
| Train job | PARAM **99003** (~5h 13m, early stop ep 30) |
| Val best | **epoch ~10** · W-F1 **0.9494** · kappa **0.4904** |
| Report ckpt | `checkpoints/run_c6_head_v0/best.pt` (on PARAM; `.pt` not in git) |
| Val JSON | [`results/run_c6_head_v0/best_metrics.json`](results/run_c6_head_v0/best_metrics.json) — tiles **31UFP + 31UGP** |
| Test JSON | [`results/run_c6_head_v0/test_metrics.json`](results/run_c6_head_v0/test_metrics.json) — tile **31UEQ** |
| P3 probes | [`results/probe_c6_v0/probe_summary_linear.md`](results/probe_c6_v0/probe_summary_linear.md) — job **99281** |

**Not done yet:** P5 full fine-tune (job **99416**) · P3 bar chart for sir · 10-class.

---

## Val best (checkpoint selection, tiles 31UFP + 31UGP)

From `best_metrics.json` at **best.pt** (epoch ~10, job **99003**).

| Class | Name | Precision | Recall | F1 |
|-------|------|-----------|--------|-----|
| 1 | Dense Built-Up | 0.2374 | 0.6602 | 0.3492 |
| 2 | Sparse Built-Up | 0.4745 | 0.6200 | 0.5376 |
| 3 | Specialized Built-Up | 0.4265 | 0.3447 | 0.3813 |
| 4 | Specialized but Vegetative | 0.0571 | 0.1398 | 0.0811 |
| 5 | Large Scale Networks | 0.1774 | 0.5876 | 0.2725 |
| 6 | Non-urban / other | 0.9920 | 0.9593 | 0.9754 |
| **W-Avg** | | **0.9632** | **0.9389** | **0.9494** |

Kappa: **0.4904** · Accuracy: **0.9389** · Mean F1: **0.4328**

**Train log (99003):** head-only P4; train loss ~0.97→0.74; val W-F1 plateaued after ep ~10; early stop ep **30** / 80 max; LR → `1e-4` after ep 16. Slurm log: `multisenge_utae/artifacts/slurm-99003.out` (PARAM only).

---

## Headline vs frozen A4 (test 31UEQ)

| Metric | U-TAE head (P4) | A4 ConvLSTM ([`RESULTS_RS2023_6CLASS.md`](../multisenge_seg/RESULTS_RS2023_6CLASS.md)) | Δ |
|--------|-----------------|----------------------------------------------------------------------------------------|---|
| **W-F1** | **0.9012** | **0.9037** | −0.0025 |
| **Kappa** | **0.4033** | **0.4424** | −0.039 |
| W-Precision | 0.9357 | 0.9559 | −0.020 |
| W-Recall / W-Sens | 0.8778 | 0.8681 | +0.010 |
| W-Specificity | 0.8592 | — | — |
| Accuracy | 0.8778 | ~0.88 | ~similar |

P4 head-only is **essentially tied** with A4 on W-F1 but **below on kappa** and on most **urban** per-class F1.

---

## Table 5 style — test (U-TAE head)

| Class | Name | Precision | Recall | Sensitivity | Specificity | F1 |
|-------|------|-----------|--------|-------------|-------------|-----|
| 1 | Dense Built-Up | 0.1688 | 0.5336 | 0.5336 | 0.9911 | 0.2565 |
| 2 | Sparse Built-Up | 0.3552 | 0.7260 | 0.7260 | 0.9586 | 0.4770 |
| 3 | Specialized Built-Up | 0.2013 | 0.2277 | 0.2277 | 0.9811 | 0.2137 |
| 4 | Specialized but Vegetative | 0.0863 | 0.0982 | 0.0982 | 0.9942 | 0.0919 |
| 5 | Large Scale Networks | 0.1524 | 0.6532 | 0.6532 | 0.9611 | 0.2472 |
| 6 | Non-urban / other | 0.9877 | 0.9055 | 0.9055 | 0.8508 | 0.9448 |
| **W-Avg** | | **0.9357** | **0.8778** | **0.8778** | **0.8592** | **0.9012** |

Kappa: **0.4033** · Mean F1: **0.3718**

---

## Per-class F1 vs A4 (test)

| Class | U-TAE head F1 | A4 F1 | Note |
|-------|---------------|-------|------|
| 1 | 0.256 | 0.489 | below A4 |
| 2 | 0.477 | 0.665 | below A4 |
| 3 | 0.214 | 0.456 | below A4 |
| 4 | 0.092 | 0.088 | similar (both low) |
| 5 | 0.247 | 0.364 | below A4 |
| 6 | 0.945 | 0.934 | **above A4** |

---

## Val vs test (head best.pt)

| Split | Tiles | W-F1 | Kappa | JSON |
|-------|-------|------|-------|------|
| Val | 31UFP + 31UGP | 0.9494 | 0.4904 | `best_metrics.json` |
| Test | 31UEQ | 0.9012 | 0.4033 | `test_metrics.json` |

Val→test gap: W-F1 **−0.048**, kappa **−0.087** (expected geographic holdout).

---

## P3 — layer probes (linear, val, job 99281)

| Level | W-F1 | Kappa |
|-------|------|-------|
| L0 | 0.7312 | 0.0809 |
| L1 | 0.7333 | 0.0810 |
| L2 | **0.7477** | 0.0772 |
| L3 | 0.4325 | 0.0126 |

Best linear probe: **L2**. L3 low (coarse L-TAE map; trained decoder does better in P4).

**Sir plot (bar chart — not an epoch curve):**
```bash
python -m multisenge_utae.plot_probe_summary \
  multisenge_utae/results/probe_c6_v0/probe_summary_linear.json \
  --title "U-TAE P3 linear probes (val)"
```

---

## Next steps

1. **P5** — job **99416** full fine-tune → test eval (target: beat A4 W-F1 **0.9037**)
2. **P3 plot** — bar chart for sir: `python -m multisenge_utae.plot_probe_summary results/probe_c6_v0/probe_summary_linear.json`
3. **Optional** — re-run P3 probes after P5 (fine-tuned encoder)
