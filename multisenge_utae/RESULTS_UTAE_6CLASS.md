# U-TAE 6-class (P4 head and P5 full)

**Model:** U-TAE, fused S2+S1 (12 channels).  
**Protocol:** Same geographic split as `multisenge_seg`. Test tile **31UEQ**. Val tiles **31UFP + 31UGP**.

| Step | Train job | Epochs ran | Best val epoch | Report checkpoint | Test JSON |
|------|-----------|------------:|---------------:|-------------------|-----------|
| P4 head | **99003** | 30 | 10 | `checkpoints/run_c6_head_v0/best.pt` | [`results/concat_utae/run_c6_head_v0/test_metrics.json`](results/concat_utae/run_c6_head_v0/test_metrics.json) |
| P5 full | **99416** | 40 | 20 | `checkpoints/run_c6_full_v0/best.pt` | [`results/concat_utae/run_c6_full_v0/test_metrics.json`](results/concat_utae/run_c6_full_v0/test_metrics.json) |

P4 freezes the encoder and the L-TAE and trains the decoder. Best val weighted F1 **0.9494**, kappa **0.4904** (job 99003, early stop at epoch 30). Val JSON: [`results/concat_utae/run_c6_head_v0/best_metrics.json`](results/concat_utae/run_c6_head_v0/best_metrics.json).

P5 loads that head `best.pt` and trains every weight. Test eval job **99628**. Best val weighted F1 **0.9585**, kappa **0.5600**. Val JSON: `checkpoints/run_c6_full_v0/best_metrics.json`.

A4 in the tables below is the reimplemented ConvLSTM+Inception row (`last.pt`, epoch 25), test weighted F1 **0.9037** / kappa **0.4424**.

---

## Headline (test 31UEQ)

| Metric | P5 full | P4 head | A4 | Δ P5 vs A4 |
|--------|--------:|--------:|---:|-----------:|
| **W-F1** | **0.9387** | 0.9012 | 0.9037 | **+0.035** |
| **Kappa** | **0.5757** | 0.4033 | 0.4424 | **+0.133** |
| W-Precision | 0.9618 | 0.9357 | 0.9559 | +0.006 |
| W-Recall / W-Sens | 0.9225 | 0.8778 | 0.8681 | +0.054 |
| W-Specificity | 0.9324 | 0.8592 | - | - |
| Accuracy | 0.9225 | 0.8778 | ~0.88 | higher |
| Mean F1 | 0.5540 | 0.3718 | - | - |

P5 is the reported 6-class row. It is above A4 on weighted F1 and on kappa.

---

## P4 head

### Val best (31UFP + 31UGP, epoch 10)

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

Train log (99003): train loss ~0.97→0.74. Validation weighted F1 peaked at epoch 10. Early stop at epoch 30 of 80. Learning rate dropped to `1e-4` after epoch 16. Slurm log: `multisenge_utae/artifacts/slurm-99003.out` (PARAM only).

### Test (31UEQ)

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

| Split | Tiles | W-F1 | Kappa |
|-------|-------|-----:|------:|
| Val | 31UFP + 31UGP | 0.9494 | 0.4904 |
| Test | 31UEQ | 0.9012 | 0.4033 |

Val to test drop: weighted F1 **0.048**, kappa **0.087**.

---

## P5 full

### Test (31UEQ)

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

| Split | W-F1 | Kappa |
|-------|-----:|------:|
| Val (31UFP+31UGP) | 0.9585 | 0.5600 |
| Test (31UEQ) | 0.9387 | 0.5757 |

---

## Per-class F1 (test)

| Class | P5 full | P4 head | A4 | Note |
|-------|--------:|--------:|---:|------|
| 1 | 0.408 | 0.256 | 0.489 | above P4, under A4 |
| 2 | **0.690** | 0.477 | 0.665 | **above A4** |
| 3 | **0.651** | 0.214 | 0.456 | **above A4** |
| 4 | 0.158 | 0.092 | 0.088 | highest of the three |
| 5 | **0.452** | 0.247 | 0.364 | **above A4** |
| 6 | **0.965** | 0.945 | 0.934 | **above A4** |

---

## P3 layer probes (linear, val, job 99281)

Checkpoint: `checkpoints/run_c6_head_v0/best.pt`. The encoder stayed frozen in P4, so these maps are the random initialization. Summary: [`results/concat_utae/probe_c6_v0/probe_summary_linear.md`](results/concat_utae/probe_c6_v0/probe_summary_linear.md).

| Level | W-F1 | Kappa |
|-------|-----:|------:|
| L0 | 0.7312 | 0.0809 |
| L1 | 0.7333 | 0.0810 |
| L2 | **0.7477** | 0.0772 |
| L3 | 0.4325 | 0.0126 |

Best linear probe: **L2**. L3 is the coarse map after the L-TAE. The trained decoder in P4 scores higher than these probes.

```bash
python -m multisenge_utae.plot_probe_summary \
  multisenge_utae/results/concat_utae/probe_c6_v0/probe_summary_linear.json \
  --title "U-TAE P3 linear probes (val)"
```
