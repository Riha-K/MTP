# U-TAE A5 — 6-class test results (P5 full fine-tune) vs A4 / P4

**Model:** U-TAE · **Phase:** P5 (`--mode full`, init from P4 head `best.pt`)  
**Protocol:** Same geographic split (test tile **31UEQ**, ~610 patches, 6-class).

| Item | Value |
|------|--------|
| Train job | PARAM **99416** · early stop ep **40** · best val W-F1 **0.9585** · kappa **0.560** |
| Report ckpt | `checkpoints/run_c6_full_v0/best.pt` (PARAM) |
| Val JSON | `checkpoints/run_c6_full_v0/best_metrics.json` |
| Test eval | job **99628** |
| Test JSON | [`results/run_c6_full_v0/test_metrics.json`](results/run_c6_full_v0/test_metrics.json) |

---

## Headline (test 31UEQ) — **beats A4**

| Metric | **P5 full** | P4 head | A4 ConvLSTM | Δ vs A4 |
|--------|-------------|---------|-------------|---------|
| **W-F1** | **0.9387** | 0.9012 | 0.9037 | **+0.035** |
| **Kappa** | **0.5757** | 0.4033 | 0.4424 | **+0.133** |
| Accuracy | 0.9225 | 0.8778 | ~0.88 | higher |
| Mean F1 | 0.5540 | 0.3718 | — | — |

---

## Table 5 style — test (U-TAE P5 full)

| Class | Name | Precision | Recall | Sens | Spec | F1 |
|-------|------|-----------|--------|------|------|-----|
| 1 | Dense Built-Up | 0.5111 | 0.3400 | 0.3400 | 0.9989 | 0.4083 |
| 2 | Sparse Built-Up | 0.5865 | 0.8367 | 0.8367 | 0.9815 | 0.6896 |
| 3 | Specialized Built-Up | 0.6882 | 0.6167 | 0.6167 | 0.9942 | 0.6505 |
| 4 | Specialized but Vegetative | 0.0925 | 0.5477 | 0.5477 | 0.9700 | 0.1582 |
| 5 | Large Scale Networks | 0.3177 | 0.7816 | 0.7816 | 0.9820 | 0.4518 |
| 6 | Non-urban / other | 0.9943 | 0.9381 | 0.9381 | 0.9284 | 0.9653 |
| **W-Avg** | | **0.9618** | **0.9225** | **0.9225** | **0.9324** | **0.9387** |

Kappa: **0.5757** · Mean F1: **0.5540**

---

## Per-class F1 — P5 vs P4 vs A4 (test)

| Class | P5 full | P4 head | A4 | Note |
|-------|---------|---------|-----|------|
| 1 | 0.408 | 0.256 | 0.489 | better than P4; still below A4 |
| 2 | **0.690** | 0.477 | 0.665 | **above A4** |
| 3 | **0.651** | 0.214 | 0.456 | **above A4** |
| 4 | 0.158 | 0.092 | 0.088 | best of three (still hard) |
| 5 | **0.452** | 0.247 | 0.364 | **above A4** |
| 6 | **0.965** | 0.945 | 0.934 | **above A4** |

---

## Val vs test (P5 best.pt)

| Split | W-F1 | Kappa |
|-------|------|-------|
| Val (31UFP+31UGP) | 0.9585 | 0.5600 |
| Test (31UEQ) | 0.9387 | 0.5757 |

---

## Status

**6-class A5 P5 frozen for sir headline** (W-F1 **0.9387** > A4 **0.9037**).  
Optional: P5 learning-curve plot · P3 bar chart · P3 re-probe on full ckpt · 10-class.
