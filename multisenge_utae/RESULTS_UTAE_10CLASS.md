# U-TAE 10-class (P4 head and P5 full)

**Model:** U-TAE, fused S2+S1 (12 channels).  
**Protocol:** Same geographic split as `multisenge_seg`. Test tile **31UEQ**. Val tiles **31UFP + 31UGP**.

| Step | Train job | Eval job | Epochs ran | Best val epoch | Report checkpoint | Test JSON |
|------|-----------|----------|------------:|---------------:|-------------------|-----------|
| P4 head | **100067** | **100433** | 54 | 34 | `checkpoints/run_c10_head_v0/best.pt` | [`results/concat_utae/run_c10_head_v0/test_metrics.json`](results/concat_utae/run_c10_head_v0/test_metrics.json) |
| P5 full | **100432** | **100503** | 42 | 22 | `checkpoints/run_c10_full_v0/best.pt` | [`results/concat_utae/run_c10_full_v0/test_metrics.json`](results/concat_utae/run_c10_full_v0/test_metrics.json) |

P4 freezes the encoder and the L-TAE and trains the decoder. Best val weighted F1 **0.8272**, kappa **0.7390** (early stop at epoch 54). Taken from `checkpoints/run_c10_head_v0/history.json`.

P5 loads that head `best.pt` and trains every weight. Best val weighted F1 **0.8625**, kappa **0.7982** (early stop at epoch 42). Taken from `checkpoints/run_c10_full_v0/history.json`.

A4 in the tables below is the reimplemented ConvLSTM+Inception row (`best.pt`, epoch 25), test weighted F1 **0.8711** / kappa **0.7588**. The printed author row is **0.8851** / **0.7945**.

---

## Headline (test 31UEQ)

| Metric | P5 full | P4 head | A4 | Paper | Δ P5 vs A4 | Δ P5 vs paper |
|--------|--------:|--------:|---:|------:|-----------:|--------------:|
| **W-F1** | **0.8811** | 0.8322 | 0.8711 | 0.8851 | **+0.010** | −0.004 |
| **Kappa** | **0.7795** | 0.6971 | 0.7588 | 0.7945 | **+0.021** | −0.015 |
| W-Precision | 0.8997 | 0.8549 | - | - | - | - |
| W-Recall / W-Sens | 0.8740 | 0.8262 | - | - | - | - |
| W-Specificity | 0.9659 | 0.9512 | - | - | - | - |
| Accuracy | 0.8740 | 0.8262 | - | - | - | - |
| Mean F1 | 0.6043 | 0.4591 | - | - | - | - |

P5 is the reported 10-class row. It is above A4. It is short of the printed author row by 0.0040 weighted F1 and 0.0150 kappa.

---

## P4 head

### Val best (31UFP + 31UGP, epoch 34)

| Split | W-F1 | Kappa | Accuracy | Mean F1 |
|-------|-----:|------:|---------:|--------:|
| Val | 0.8272 | 0.7390 | 0.8143 | 0.4718 |

Per-class validation JSON is not in `results/concat_utae/run_c10_head_v0/`. The scores above are the best row of `history.json`.

### Test (31UEQ)

| Class | Name | Precision | Recall | Sensitivity | Specificity | F1 |
|-------|------|-----------|--------|-------------|-------------|-----|
| 1 | Dense Built-Up | 0.2229 | 0.5141 | 0.5141 | 0.9939 | 0.3110 |
| 2 | Sparse Built-Up | 0.4696 | 0.7249 | 0.7249 | 0.9743 | 0.5700 |
| 3 | Specialized Built-Up Areas | 0.5981 | 0.0930 | 0.0930 | 0.9987 | 0.1610 |
| 4 | Specialized but Vegetative Areas | 0.0401 | 0.1378 | 0.1378 | 0.9816 | 0.0621 |
| 5 | Large Scale Networks | 0.2243 | 0.6240 | 0.6240 | 0.9769 | 0.3300 |
| 6 | Arable Lands | 0.9623 | 0.9357 | 0.9357 | 0.9364 | 0.9488 |
| 7 | Vineyards and Orchards | 0.5624 | 0.4624 | 0.4624 | 0.9777 | 0.5076 |
| 8 | Grasslands | 0.5910 | 0.3894 | 0.3894 | 0.9823 | 0.4695 |
| 9 | Forests and semi-natural areas | 0.8581 | 0.8575 | 0.8575 | 0.9716 | 0.8578 |
| 10 | Water Surfaces | 0.2603 | 0.6612 | 0.6612 | 0.9840 | 0.3736 |
| **W-Avg** | | **0.8549** | **0.8262** | **0.8262** | **0.9512** | **0.8322** |

Kappa: **0.6971** · Accuracy: **0.8262** · Mean F1: **0.4591**

| Split | W-F1 | Kappa |
|-------|-----:|------:|
| Val (31UFP+31UGP) | 0.8272 | 0.7390 |
| Test (31UEQ) | 0.8322 | 0.6971 |

---

## P5 full

### Val best (31UFP + 31UGP, epoch 22)

| Split | W-F1 | Kappa | Accuracy | Mean F1 |
|-------|-----:|------:|---------:|--------:|
| Val | 0.8625 | 0.7982 | 0.8579 | 0.5984 |

### Test (31UEQ)

| Class | Name | Precision | Recall | Sens | Spec | F1 |
|-------|------|-----------|--------|------|------|-----|
| 1 | Dense Built-Up | 0.4260 | 0.5455 | 0.5455 | 0.9975 | 0.4784 |
| 2 | Sparse Built-Up | 0.6422 | 0.8215 | 0.8215 | 0.9856 | 0.7209 |
| 3 | Specialized Built-Up Areas | 0.4656 | 0.7708 | 0.7708 | 0.9815 | 0.5806 |
| 4 | Specialized but Vegetative Areas | 0.1076 | 0.2997 | 0.2997 | 0.9861 | 0.1584 |
| 5 | Large Scale Networks | 0.3652 | 0.7606 | 0.7606 | 0.9859 | 0.4935 |
| 6 | Arable Lands | 0.9736 | 0.9552 | 0.9552 | 0.9552 | 0.9644 |
| 7 | Vineyards and Orchards | 0.8973 | 0.6831 | 0.6831 | 0.9952 | 0.7757 |
| 8 | Grasslands | 0.7088 | 0.4443 | 0.4443 | 0.9880 | 0.5462 |
| 9 | Forests and semi-natural areas | 0.8894 | 0.8508 | 0.8508 | 0.9788 | 0.8697 |
| 10 | Water Surfaces | 0.3222 | 0.7778 | 0.7778 | 0.9861 | 0.4557 |
| **W-Avg** | | **0.8997** | **0.8740** | **0.8740** | **0.9659** | **0.8811** |

Kappa: **0.7795** · Accuracy: **0.8740** · Mean F1: **0.6043**

| Split | W-F1 | Kappa |
|-------|-----:|------:|
| Val (31UFP+31UGP) | 0.8625 | 0.7982 |
| Test (31UEQ) | 0.8811 | 0.7795 |

---

## Per-class F1 (test)

| Class | P5 full | P4 head | A4 | Paper | Note |
|-------|--------:|--------:|---:|------:|------|
| 1 Dense built-up | 0.478 | 0.311 | 0.442 | 0.503 | above A4, under the paper |
| 2 Sparse built-up | 0.721 | 0.570 | 0.687 | 0.730 | above A4, under the paper |
| 3 Specialized built-up | **0.581** | 0.161 | 0.506 | 0.575 | **above A4 and the paper** |
| 4 Specialized vegetative | 0.158 | 0.062 | 0.171 | 0.247 | under A4 |
| 5 Networks | 0.494 | 0.330 | 0.469 | 0.547 | above A4, under the paper |
| 6 Arable | **0.964** | 0.949 | 0.960 | 0.964 | **above A4**, level with the paper |
| 7 Vineyards and orchards | 0.776 | 0.508 | 0.822 | 0.869 | under A4 |
| 8 Grasslands | **0.546** | 0.470 | 0.520 | 0.516 | **above A4 and the paper** |
| 9 Forest and semi-natural | **0.870** | 0.858 | 0.839 | 0.861 | **above A4 and the paper** |
| 10 Water | 0.456 | 0.374 | 0.437 | 0.561 | above A4, under the paper |

---

## P3 layer probes (linear, val, job 100505)

`probe_c10.sbatch` defaults to `checkpoints/run_c10_head_v0/best.pt`. The summary files do not store the checkpoint path. Summary: [`results/concat_utae/probe_c10_v0/probe_summary_linear.md`](results/concat_utae/probe_c10_v0/probe_summary_linear.md).

| Level | W-F1 | Kappa |
|-------|-----:|------:|
| L0 | 0.7948 | 0.6643 |
| L1 | **0.8009** | 0.6885 |
| L2 | 0.7040 | 0.5646 |
| L3 | 0.5335 | 0.3438 |

Best linear probe: **L1**. L3 is the weakest of the four maps.
