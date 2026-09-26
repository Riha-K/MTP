# MultiSenGE living results board

Single place for **paper (first-author)**, **A4 ConvLSTM**, **concat U-TAE**, **MA-UTAE**, and modality ablations. All test numbers = tile **31UEQ** (geographic split, 4-date S1+S2 unless noted). Metrics from shared `multisenge_seg/metrics.py` (P / R / Sens / Spec / F1 / W-* / Kappa).

**Legend:** `[ok]` on laptop | `[head]` headline only (JSON still on PARAM) | `[todo]` not run / missing | `~(val)` = validation only

**Last updated:** 2026-09-23

---

## 0. Inventory - what is on this laptop

| Run | test_metrics.json | history_plot.png | Notes |
|-----|:-----------------:|:----------------:|-------|
| Paper 6c (Table 5/6) | - | - | Transcribed — [`PAPER_MODALITY_6CLASS.md`](PAPER_MODALITY_6CLASS.md) |
| Paper 10c (Table 7/8) | - | - | Transcribed — S1 W-F1 **0.8055** / κ **0.6422**; S2 **0.8696** / **0.7445**; Inc **0.8851** / **0.7945** |
| A4 6c last.pt (report) | [ok] `multisenge_seg/results/run_c6_v0/last_test_metrics.json` | - | Report row |
| A4 6c best.pt | [ok] `.../test_metrics.json` | - | Not the report row |
| A4 10c best.pt | [ok] `multisenge_seg/results/run_c10_v0/test_metrics.json` | - |  |
| Concat U-TAE 6c P4 | [ok] | [ok] | `results/concat_utae/run_c6_head_v0/` |
| Concat U-TAE 6c P5 | [ok] | [ok] | `results/concat_utae/run_c6_full_v0/` |
| Concat U-TAE 10c P4 | [ok] | [ok] | `results/concat_utae/run_c10_head_v0/` |
| Concat U-TAE 10c P5 | [ok] | [ok] | `results/concat_utae/run_c10_full_v0/` |
| MA gated 6c P4 | [ok] | [ok] | `results/ma_utae/ma_c6_gated_head_v0/` |
| MA gated 6c P5 | [ok] | [ok] | `results/ma_utae/ma_c6_gated_full_v0/` |
| S2-only 6c P4 | [ok] | [ok] | `results/concat_utae/run_c6_s2_head_v0/` |
| S1-only 6c P4 | [ok] | [ok] | `results/concat_utae/run_c6_s1_head_v0/` |
| S1-only 6c P5 | [ok] | [ok] | W-F1 0.8970 / κ 0.3537 |
| S2-only 6c P5 | [ok] | [todo] | W-F1 **0.9199** / κ **0.4809** (JSON may be on PARAM) |
| MA concat 6c P4 | [ok] | [todo] | W-F1 **0.9218** / κ **0.4961** |
| MA concat 6c P5 | [ok] | [todo] | W-F1 **0.9143** / κ **0.4672** |
| **MA gated 10c P4** | [ok] | [ok] | W-F1 **0.8430** / κ **0.7061** |
| **MA gated 10c P5** | [ok] | [ok] | W-F1 **0.8834** / κ **0.7895** — near paper |
| **MA concat 10c P4** | [ok] | [ok]? | W-F1 **0.8547** / κ **0.7273** (plot if present) |
| **MA concat 10c P5** | [ok] | [ok] | W-F1 **0.8885** / κ **0.7950** — **beats paper Inc** (+0.0034 / +0.0005) |
| **S1-only 10c P4** | [ok] | [no plot] | W-F1 **0.7342** / κ **0.5396** (history plot missing on PARAM) |
| **S2-only 10c P4** | [ok] | [ok] | W-F1 **0.8437** / κ **0.7152** |
| **S1-only 10c P5** | [ok] | [ok] | W-F1 **0.8365** / κ **0.6939** — beats paper S1 (+0.031 / +0.052) |
| **S2-only 10c P5** | [ok] | [ok] | W-F1 **0.8865** / κ **0.7945** — beats paper S2 (+0.017 / +0.050) |
| Task H | [deferred] | — | **Future work** — not needed for Task M write-up |
| P3 probes 6c/10c | [ok] summaries | [ok] | `results/concat_utae/probe_c{6,10}_v0/` |

### Status (2026-09-23)

**Task M closed** on 6c + 10c (fusion + S1/S2 ablations).  
**Task H (hierarchy A1+A2):** **skip / defer** — optional future work on MA concat 10c if we want extra UF / Dense↔Sparse gains. Not required to claim Task M.

**Paper-facing one-winner lanes** (full tables → §1.3):

| Lane | Report this one | W-F1 / κ | vs fair paper |
|------|-----------------|----------|---------------|
| **6c fusion** | Stock **Concat U-TAE P5** | **0.9387 / 0.5757** | Inc **+0.037 / +0.157** |
| **6c best MA** | **MA gated P5** | **0.9353 / 0.5520** | Inc +0.034 / +0.133 (still under stock concat) |
| **6c S1** | U-TAE **S1 P4** | **0.9111 / 0.3974** | ConvLSTM-S1 **+0.011 / +0.005** |
| **6c S2** | U-TAE **S2 P5** | **0.9199 / 0.4809** | ConvLSTM-S2 **+0.024 / +0.059** |
| **10c fusion** | **MA concat P5** | **0.8885 / 0.7950** | Inc **+0.003 / +0.001** |
| **10c S1** | U-TAE **S1 P5** | **0.8365 / 0.6939** | ConvLSTM-S1 **+0.031 / +0.052** |
| **10c S2** | U-TAE **S2 P5** | **0.8865 / 0.7945** | ConvLSTM-S2 **+0.017 / +0.050** |

---
## 1. Headline comparison (paper-facing)

### 1.1 Six-class test (31UEQ)

| Rank | Model | Phase | W-P | W-R | **W-F1** | **Kappa** | Acc | Mean F1 | vs concat P5 W-F1 |
|-----:|-------|-------|----:|----:|---------:|----------:|----:|--------:|------------------:|
| 1 | Concat U-TAE | P5 | 0.9618 | 0.9225 | **0.9387** | **0.5757** | 0.9225 | 0.5540 | - |
| 2 | MA gated | P5 | 0.9607 | 0.9165 | **0.9353** | **0.5520** | 0.9165 | 0.5384 | -0.0034 |
| 3 | MA concat fuse | P4 | 0.9473 | 0.9048 | **0.9218** | **0.4961** | 0.9048 | 0.4606 | -0.0169 |
| 4 | S2-only U-TAE | P5 | 0.9432 | 0.9069 | **0.9199** | **0.4809** | 0.9069 | 0.4261 | -0.0188 |
| 5 | S2-only U-TAE | P4 | 0.9482 | 0.8960 | **0.9171** | **0.4658** | 0.8960 | 0.4617 | -0.0216 (head) |
| 6 | MA gated | P4 | 0.9478 | 0.8954 | **0.9169** | **0.4705** | 0.8954 | 0.4573 | -0.0218 |
| 7 | MA concat fuse | P5 | 0.9502 | 0.8910 | **0.9143** | **0.4672** | 0.8910 | 0.4667 | -0.0244 |
| 8 | S1-only U-TAE | P4 | 0.9320 | 0.9026 | **0.9111** | **0.3974** | 0.9026 | 0.3100 | -0.0276 |
| 9 | A4 ConvLSTM (report last.pt) | - | 0.9559 | 0.8681 | **0.9037** | **0.4424** | 0.8681 | - | -0.0350 |
| 10 | **Paper** ConvLSTM+Inception | - | 0.9591 | 0.8596 | **0.9018** | **0.4186** | - | - | -0.0369 |
| 11 | Concat U-TAE | P4 | 0.9357 | 0.8778 | **0.9012** | **0.4033** | 0.8778 | 0.3718 | -0.0375 |
| 12 | S1-only U-TAE | P5 | 0.9395 | 0.8715 | **0.8970** | **0.3537** | 0.8715 | 0.3168 | -0.0417 (worse than S1 P4) |

### 1.2 Ten-class test (31UEQ)

| Rank | Model | Phase | W-P | W-R | **W-F1** | **Kappa** | Acc | Mean F1 | note |
|-----:|-------|-------|----:|----:|---------:|----------:|----:|--------:|------|
| 1 | **MA concat fuse** | **P5** | 0.9040 | 0.8817 | **0.8885** | **0.7950** | 0.8817 | 0.6332 | vs Inc **+0.0034 / +0.0005** |
| 2 | **S2-only U-TAE** | **P5** | 0.9005 | 0.8819 | **0.8865** | **0.7945** | 0.8819 | 0.6230 | vs paper **S2** **+0.017 / +0.050** |
| 3 | **Paper** ConvLSTM+Inception | - | 0.8977 | 0.8831 | **0.8851** | **0.7945** | - | - | fusion baseline |
| 4 | **MA gated** | **P5** | 0.8994 | 0.8789 | **0.8834** | **0.7895** | 0.8789 | - | vs Inc −0.0017 / −0.0050 |
| 5 | Concat U-TAE | P5 | 0.8997 | 0.8740 | **0.8811** | **0.7795** | 0.8740 | 0.6043 | vs Inc −0.0040 |
| 6 | A4 ConvLSTM | - | 0.8947 | 0.8604 | **0.8711** | **0.7588** | 0.8604 | 0.5853 | vs Inc −0.0140 |
| 7 | **Paper** ConvLSTM-S2 | - | 0.9000 | 0.8517 | **0.8696** | **0.7445** | - | - | fair S2 baseline |
| 8 | MA concat fuse | P4 | 0.8790 | 0.8425 | **0.8547** | **0.7273** | 0.8425 | - | |
| 9 | S2-only U-TAE | P4 | 0.8643 | 0.8365 | **0.8437** | **0.7152** | 0.8365 | 0.5150 | |
| 10 | MA gated | P4 | 0.8735 | 0.8272 | **0.8430** | **0.7061** | 0.8272 | 0.5111 | |
| 11 | **S1-only U-TAE** | **P5** | 0.8607 | 0.8207 | **0.8365** | **0.6939** | 0.8207 | 0.5157 | vs paper **S1** **+0.031 / +0.052** |
| 12 | Concat U-TAE | P4 | 0.8549 | 0.8262 | **0.8322** | **0.6971** | 0.8262 | 0.4591 | |
| 13 | **Paper** ConvLSTM-S1 | - | 0.8422 | 0.7836 | **0.8055** | **0.6422** | - | - | fair S1 baseline |
| 14 | S1-only U-TAE | P4 | 0.7792 | 0.7146 | **0.7342** | **0.5396** | 0.7146 | 0.3418 | |

_MA concat P5 from PARAM eval **104545**; JSON+plot on laptop `results/ma_utae/ma_c10_concat_full_v0/`. S1/S2 P5 on laptop. Fair S2 claim is vs ConvLSTM-S2 only; do not sell S2 as beating paper Inc._

### 1.3 Deltas vs fair paper row (6c alone / 10c alone)

#### 6c — MA gated / concat vs paper Inc (0.9018 / 0.4186)

| Model | Phase | W-F1 | κ | Δ W-F1 | Δ κ |
|-------|-------|-----:|--:|-------:|----:|
| Paper Inc | — | 0.9018 | 0.4186 | — | — |
| Stock Concat U-TAE | **P5** | **0.9387** | **0.5757** | **+0.0369** | **+0.1571** |
| MA gated | P5 | 0.9353 | 0.5520 | +0.0335 | +0.1334 |
| MA gated | P4 | 0.9169 | 0.4705 | +0.0151 | +0.0519 |
| MA concat | P4 | 0.9218 | 0.4961 | +0.0200 | +0.0775 |
| MA concat | P5 | 0.9143 | 0.4672 | +0.0125 | +0.0486 |

**6c S1** vs paper S1 (0.9001 / 0.3929): P4 **+0.011 / +0.005** · P5 **−0.003 / −0.039**  
**6c S2** vs paper S2 (0.8958 / 0.4223): P4 **+0.021 / +0.044** · P5 **+0.024 / +0.059**

#### 10c — MA gated / concat vs paper Inc (0.8851 / 0.7945)

| Model | Phase | W-F1 | κ | Δ W-F1 | Δ κ |
|-------|-------|-----:|--:|-------:|----:|
| Paper Inc | — | 0.8851 | 0.7945 | — | — |
| **MA concat** | **P5** | **0.8885** | **0.7950** | **+0.0034** | **+0.0005** |
| MA gated | P5 | 0.8834 | 0.7895 | −0.0017 | −0.0050 |
| Stock Concat U-TAE | P5 | 0.8811 | 0.7795 | −0.0040 | −0.0150 |
| MA concat | P4 | 0.8547 | 0.7273 | −0.0304 | −0.0672 |
| MA gated | P4 | 0.8430 | 0.7061 | −0.0421 | −0.0884 |

**10c S1** vs paper S1 (0.8055 / 0.6422): P4 **−0.071 / −0.103** · P5 **+0.031 / +0.052**  
**10c S2** vs paper S2 (0.8696 / 0.7445): P4 **−0.026 / −0.029** · P5 **+0.017 / +0.050**

---

## 2. Six-class - full globals + per-class (everything)

### 2.1 Global metrics

| Model | Acc | W-P | W-R | W-Sens | W-Spec | **W-F1** | Mean F1 | Mean Sens | Mean Spec | **Kappa** |
|-------|----:|----:|----:|-------:|-------:|---------:|--------:|----------:|----------:|----------:|
| **Paper** ConvLSTM+Inception | - | 0.9591 | 0.8596 | 0.8596 | - | **0.9018** | - | - | - | **0.4186** |
| A4 report `last.pt` ep25 | 0.8681 | 0.9559 | 0.8681 | 0.8681 | - | **0.9037** | - | - | - | **0.4424** |
| A4 `best.pt` (not report) | 0.8808 | 0.9506 | 0.8808 | 0.8808 | - | **0.9098** | 0.4743 | - | - | **0.4496** |
| Concat U-TAE P4 head | 0.8778 | 0.9357 | 0.8778 | 0.8778 | 0.8592 | **0.9012** | 0.3718 | 0.5240 | 0.9562 | **0.4033** |
| Concat U-TAE P5 full | 0.9225 | 0.9618 | 0.9225 | 0.9225 | 0.9324 | **0.9387** | 0.5540 | 0.6768 | 0.9758 | **0.5757** |
| MA gated P4 head | 0.8954 | 0.9478 | 0.8954 | 0.8954 | 0.8946 | **0.9169** | 0.4573 | 0.6208 | 0.9650 | **0.4705** |
| MA gated P5 full | 0.9165 | 0.9607 | 0.9165 | 0.9165 | 0.9213 | **0.9353** | 0.5384 | 0.6629 | 0.9730 | **0.5520** |
| S2-only P4 test | 0.8960 | 0.9482 | 0.8960 | 0.8960 | 0.8767 | **0.9171** | 0.4617 | 0.5988 | 0.9621 | **0.4658** |
| S1-only P4 test | 0.9026 | 0.9320 | 0.9026 | 0.9026 | 0.7252 | **0.9111** | 0.3100 | 0.3933 | 0.9380 | **0.3974** |
| S1-only P5 test | 0.8715 | 0.9395 | 0.8715 | 0.8715 | 0.7992 | **0.8970** | 0.3168 | 0.4000 | 0.9451 | **0.3537** |
| S1-only P5 ~(val) best | 0.9134 | 0.9605 | 0.9134 | 0.9134 | 0.7721 | **0.9322** | 0.3364 | 0.3925 | 0.9476 | **0.3601** |
| S2-only P5 test | 0.9069 | 0.9432 | 0.9069 | 0.9069 | 0.8324 | **0.9199** | 0.4261 | 0.5237 | 0.9565 | **0.4809** |
| MA concat P4 | 0.9048 | 0.9473 | 0.9048 | 0.9048 | 0.8927 | **0.9218** | 0.4606 | 0.6125 | 0.9663 | **0.4961** |
| MA concat P5 | 0.8910 | 0.9502 | 0.8910 | 0.8910 | 0.8977 | **0.9143** | 0.4667 | 0.5952 | 0.9648 | **0.4672** |

### 2.2 Per-class Precision / Recall / Sens / Spec / F1

#### Paper ConvLSTM+Inception (RS 2023 Table 5)

| Class | Name | Precision | Recall | Sensitivity | Specificity | F1 | Support |
|------:|------|----------:|-------:|------------:|------------:|---:|--------:|
| 1 | Dense Built-Up | 0.2308 | 0.8599 | 0.8599 | - | 0.3639 | - |
| 2 | Sparse Built-Up | 0.6260 | 0.6472 | 0.6472 | - | 0.6364 | - |
| 3 | Specialized Built-Up | 0.4794 | 0.7647 | 0.7647 | - | 0.5894 | - |
| 4 | Specialized but Vegetative | 0.0312 | 0.4461 | 0.4461 | - | 0.0584 | - |
| 5 | Large Scale Networks | 0.2736 | 0.7898 | 0.7898 | - | 0.4064 | - |
| 6 | Non-urban / other | 0.9965 | 0.8719 | 0.8719 | - | 0.9301 | - |
| **W-Avg** | | 0.9591 | 0.8596 | 0.8596 | - | **0.9018** | |

Kappa 0.4186

#### A4 ConvLSTM - report `last.pt` (no Spec in JSON)

| Class | Name | Precision | Recall | Sensitivity | Specificity | F1 | Support |
|------:|------|----------:|-------:|------------:|------------:|---:|--------:|
| 1 | Dense Built-Up | 0.4118 | 0.6012 | 0.6012 | - | 0.4888 | - |
| 2 | Sparse Built-Up | 0.5851 | 0.7698 | 0.7698 | - | 0.6649 | - |
| 3 | Specialized Built-Up | 0.3264 | 0.7555 | 0.7555 | - | 0.4558 | - |
| 4 | Specialized but Vegetative | 0.0484 | 0.4680 | 0.4680 | - | 0.0877 | - |
| 5 | Large Scale Networks | 0.2359 | 0.7973 | 0.7973 | - | 0.3640 | - |
| 6 | Non-urban / other | 0.9976 | 0.8779 | 0.8779 | - | 0.9339 | - |
| **W-Avg** | | 0.9559 | 0.8681 | 0.8681 | - | **0.9037** | |

Acc 0.8681 | Kappa 0.4424

#### A4 ConvLSTM - `best.pt` (alt; do not quote as main)

| Class | Name | Precision | Recall | Sensitivity | Specificity | F1 | Support |
|------:|------|----------:|-------:|------------:|------------:|---:|--------:|
| 1 | Dense Built-Up | 0.3184 | 0.7200 | 0.7200 | - | 0.4416 | - |
| 2 | Sparse Built-Up | 0.4720 | 0.7649 | 0.7649 | - | 0.5838 | - |
| 3 | Specialized Built-Up | 0.4300 | 0.5021 | 0.5021 | - | 0.4633 | - |
| 4 | Specialized but Vegetative | 0.0388 | 0.2837 | 0.2837 | - | 0.0682 | - |
| 5 | Large Scale Networks | 0.2201 | 0.8015 | 0.8015 | - | 0.3453 | - |
| 6 | Non-urban / other | 0.9938 | 0.8980 | 0.8980 | - | 0.9435 | - |
| **W-Avg** | | 0.9506 | 0.8808 | 0.8808 | - | **0.9098** | |

Acc 0.8808 | Kappa 0.4496 | Mean F1 0.4743

#### Concat U-TAE P4 head

| Class | Name | Precision | Recall | Sensitivity | Specificity | F1 | Support |
|------:|------|----------:|-------:|------------:|------------:|---:|--------:|
| 1 | Dense Built-Up | 0.1688 | 0.5336 | 0.5336 | 0.9911 | 0.2565 | 135479 |
| 2 | Sparse Built-Up | 0.3552 | 0.7260 | 0.7260 | 0.9586 | 0.4770 | 1216210 |
| 3 | Specialized Built-Up | 0.2013 | 0.2277 | 0.2277 | 0.9811 | 0.2137 | 818189 |
| 4 | Specialized but Vegetative | 0.0863 | 0.0982 | 0.0982 | 0.9942 | 0.0919 | 222108 |
| 5 | Large Scale Networks | 0.1524 | 0.6532 | 0.6532 | 0.9611 | 0.2472 | 423239 |
| 6 | Non-urban / other | 0.9877 | 0.9055 | 0.9055 | 0.8508 | 0.9448 | 37161735 |
| **W-Avg** | | 0.9357 | 0.8778 | 0.8778 | 0.8592 | **0.9012** | |

Acc 0.8778 | Kappa 0.4033 | Mean F1 0.3718

#### Concat U-TAE P5 full

| Class | Name | Precision | Recall | Sensitivity | Specificity | F1 | Support |
|------:|------|----------:|-------:|------------:|------------:|---:|--------:|
| 1 | Dense Built-Up | 0.5111 | 0.3400 | 0.3400 | 0.9989 | 0.4083 | 135479 |
| 2 | Sparse Built-Up | 0.5865 | 0.8367 | 0.8367 | 0.9815 | 0.6896 | 1216210 |
| 3 | Specialized Built-Up | 0.6882 | 0.6167 | 0.6167 | 0.9942 | 0.6505 | 818189 |
| 4 | Specialized but Vegetative | 0.0925 | 0.5477 | 0.5477 | 0.9700 | 0.1582 | 222108 |
| 5 | Large Scale Networks | 0.3177 | 0.7816 | 0.7816 | 0.9820 | 0.4518 | 423239 |
| 6 | Non-urban / other | 0.9943 | 0.9381 | 0.9381 | 0.9284 | 0.9653 | 37161735 |
| **W-Avg** | | 0.9618 | 0.9225 | 0.9225 | 0.9324 | **0.9387** | |

Acc 0.9225 | Kappa 0.5757 | Mean F1 0.5540

#### MA gated P4 head

| Class | Name | Precision | Recall | Sensitivity | Specificity | F1 | Support |
|------:|------|----------:|-------:|------------:|------------:|---:|--------:|
| 1 | Dense Built-Up | 0.2732 | 0.4956 | 0.4956 | 0.9955 | 0.3522 | 135479 |
| 2 | Sparse Built-Up | 0.5155 | 0.7493 | 0.7493 | 0.9779 | 0.6108 | 1216210 |
| 3 | Specialized Built-Up | 0.3742 | 0.3618 | 0.3618 | 0.9874 | 0.3679 | 818189 |
| 4 | Specialized but Vegetative | 0.1162 | 0.3931 | 0.3931 | 0.9833 | 0.1794 | 222108 |
| 5 | Large Scale Networks | 0.1698 | 0.8079 | 0.8079 | 0.9577 | 0.2806 | 423239 |
| 6 | Non-urban / other | 0.9909 | 0.9174 | 0.9174 | 0.8883 | 0.9527 | 37161735 |
| **W-Avg** | | 0.9478 | 0.8954 | 0.8954 | 0.8946 | **0.9169** | |

Acc 0.8954 | Kappa 0.4705 | Mean F1 0.4573

#### MA gated P5 full

| Class | Name | Precision | Recall | Sensitivity | Specificity | F1 | Support |
|------:|------|----------:|-------:|------------:|------------:|---:|--------:|
| 1 | Dense Built-Up | 0.4912 | 0.3193 | 0.3193 | 0.9989 | 0.3870 | 135479 |
| 2 | Sparse Built-Up | 0.6731 | 0.7691 | 0.7691 | 0.9883 | 0.7179 | 1216210 |
| 3 | Specialized Built-Up | 0.5823 | 0.6651 | 0.6651 | 0.9900 | 0.6209 | 818189 |
| 4 | Specialized but Vegetative | 0.0750 | 0.4812 | 0.4812 | 0.9668 | 0.1298 | 222108 |
| 5 | Large Scale Networks | 0.2771 | 0.8096 | 0.8096 | 0.9774 | 0.4129 | 423239 |
| 6 | Non-urban / other | 0.9933 | 0.9329 | 0.9329 | 0.9163 | 0.9621 | 37161735 |
| **W-Avg** | | 0.9607 | 0.9165 | 0.9165 | 0.9213 | **0.9353** | |

Acc 0.9165 | Kappa 0.5520 | Mean F1 0.5384

#### S2-only P4 test

| Class | Name | Precision | Recall | Sensitivity | Specificity | F1 | Support |
|------:|------|----------:|-------:|------------:|------------:|---:|--------:|
| 1 | Dense Built-Up | 0.4131 | 0.4107 | 0.4107 | 0.9980 | 0.4119 | 135479 |
| 2 | Sparse Built-Up | 0.5028 | 0.7680 | 0.7680 | 0.9762 | 0.6077 | 1216210 |
| 3 | Specialized Built-Up | 0.4610 | 0.2963 | 0.2963 | 0.9928 | 0.3607 | 818189 |
| 4 | Specialized but Vegetative | 0.0774 | 0.4091 | 0.4091 | 0.9727 | 0.1301 | 222108 |
| 5 | Large Scale Networks | 0.1904 | 0.7894 | 0.7894 | 0.9641 | 0.3068 | 423239 |
| 6 | Non-urban / other | 0.9893 | 0.9192 | 0.9192 | 0.8689 | 0.9530 | 37161735 |
| **W-Avg** | | 0.9482 | 0.8960 | 0.8960 | 0.8767 | **0.9171** | |

Acc 0.8960 | Kappa 0.4658 | Mean F1 0.4617

#### S2-only P5 test

| Class | Name | Precision | Recall | Sensitivity | Specificity | F1 | Support |
|------:|------|----------:|-------:|------------:|------------:|---:|--------:|
| 1 | Dense Built-Up | 0.4118 | 0.2481 | 0.2481 | 0.9988 | 0.3097 | 135479 |
| 2 | Sparse Built-Up | 0.4502 | 0.8262 | 0.8262 | 0.9683 | 0.5828 | 1216210 |
| 3 | Specialized Built-Up | 0.4521 | 0.2106 | 0.2106 | 0.9947 | 0.2873 | 818189 |
| 4 | Specialized but Vegetative | 0.0675 | 0.1552 | 0.1552 | 0.9880 | 0.0941 | 222108 |
| 5 | Large Scale Networks | 0.2052 | 0.7689 | 0.7689 | 0.9681 | 0.3240 | 423239 |
| 6 | Non-urban / other | 0.9857 | 0.9333 | 0.9333 | 0.8213 | 0.9588 | 37161735 |
| **W-Avg** | | 0.9432 | 0.9069 | 0.9069 | 0.8324 | **0.9199** | |

Acc 0.9069 | Kappa 0.4809 | Mean F1 0.4261

#### S1-only P4 test

| Class | Name | Precision | Recall | Sensitivity | Specificity | F1 | Support |
|------:|------|----------:|-------:|------------:|------------:|---:|--------:|
| 1 | Dense Built-Up | 0.1384 | 0.5235 | 0.5235 | 0.9889 | 0.2189 | 135479 |
| 2 | Sparse Built-Up | 0.3416 | 0.5909 | 0.5909 | 0.9643 | 0.4329 | 1216210 |
| 3 | Specialized Built-Up | 0.5811 | 0.0714 | 0.0714 | 0.9989 | 0.1272 | 818189 |
| 4 | Specialized but Vegetative | 0.0580 | 0.0047 | 0.0047 | 0.9996 | 0.0086 | 222108 |
| 5 | Large Scale Networks | 0.0743 | 0.2236 | 0.2236 | 0.9702 | 0.1115 | 423239 |
| 6 | Non-urban / other | 0.9770 | 0.9456 | 0.9456 | 0.7060 | 0.9610 | 37161735 |
| **W-Avg** | | 0.9320 | 0.9026 | 0.9026 | 0.7252 | **0.9111** | |

Acc 0.9026 | Kappa 0.3974 | Mean F1 0.3100

#### S1-only P5 test

| Class | Name | Precision | Recall | Sensitivity | Specificity | F1 | Support |
|------:|------|----------:|-------:|------------:|------------:|---:|--------:|
| 1 | Dense Built-Up | 0.1630 | 0.3397 | 0.3397 | 0.9941 | 0.2203 | 135479 |
| 2 | Sparse Built-Up | 0.3874 | 0.6667 | 0.6667 | 0.9669 | 0.4900 | 1216210 |
| 3 | Specialized Built-Up | 0.6418 | 0.0809 | 0.0809 | 0.9991 | 0.1436 | 818189 |
| 4 | Specialized but Vegetative | 0.0399 | 0.0041 | 0.0041 | 0.9995 | 0.0074 | 222108 |
| 5 | Large Scale Networks | 0.0543 | 0.4008 | 0.4008 | 0.9253 | 0.0956 | 423239 |
| 6 | Non-urban / other | 0.9825 | 0.9081 | 0.9081 | 0.7860 | 0.9438 | 37161735 |
| **W-Avg** | | 0.9395 | 0.8715 | 0.8715 | 0.7992 | **0.8970** | |

Acc 0.8715 | Kappa 0.3537 | Mean F1 0.3168

#### S1-only P5 ~(val) best (not test)

| Class | Name | Precision | Recall | Sensitivity | Specificity | F1 | Support |
|------:|------|----------:|-------:|------------:|------------:|---:|--------:|
| 1 | Dense Built-Up | 0.3265 | 0.2835 | 0.2835 | 0.9977 | 0.3035 | 488886 |
| 2 | Sparse Built-Up | 0.4286 | 0.6585 | 0.6585 | 0.9807 | 0.5192 | 2675583 |
| 3 | Specialized Built-Up | 0.6794 | 0.0779 | 0.0779 | 0.9996 | 0.1398 | 1408122 |
| 4 | Specialized but Vegetative | 0.0438 | 0.0051 | 0.0051 | 0.9997 | 0.0091 | 373839 |
| 5 | Large Scale Networks | 0.0472 | 0.3921 | 0.3921 | 0.9460 | 0.0843 | 844044 |
| 6 | Non-urban / other | 0.9878 | 0.9382 | 0.9382 | 0.7618 | 0.9624 | 118793462 |
| **W-Avg** | | 0.9605 | 0.9134 | 0.9134 | 0.7721 | **0.9322** | |

Acc 0.9134 | Kappa 0.3601 | Mean F1 0.3364

#### MA concat P4 test

| Class | Name | Precision | Recall | Sensitivity | Specificity | F1 | Support |
|------:|------|----------:|-------:|------------:|------------:|---:|--------:|
| 1 | Dense Built-Up | 0.2715 | 0.6882 | 0.6882 | 0.9937 | 0.3894 | 135479 |
| 2 | Sparse Built-Up | 0.4611 | 0.7741 | 0.7741 | 0.9716 | 0.5779 | 1216210 |
| 3 | Specialized Built-Up | 0.4304 | 0.4176 | 0.4176 | 0.9885 | 0.4239 | 818189 |
| 4 | Specialized but Vegetative | 0.1337 | 0.1372 | 0.1372 | 0.9950 | 0.1354 | 222108 |
| 5 | Large Scale Networks | 0.1724 | 0.7305 | 0.7305 | 0.9625 | 0.2790 | 423239 |
| 6 | Non-urban / other | 0.9908 | 0.9272 | 0.9272 | 0.8863 | 0.9579 | 37161735 |
| **W-Avg** | | 0.9473 | 0.9048 | 0.9048 | 0.8927 | **0.9218** | |

Acc 0.9048 | Kappa 0.4961 | Mean F1 0.4606

#### MA concat P5 test

| Class | Name | Precision | Recall | Sensitivity | Specificity | F1 | Support |
|------:|------|----------:|-------:|------------:|------------:|---:|--------:|
| 1 | Dense Built-Up | 0.6011 | 0.2246 | 0.2246 | 0.9995 | 0.3271 | 135479 |
| 2 | Sparse Built-Up | 0.4256 | 0.8667 | 0.8667 | 0.9633 | 0.5709 | 1216210 |
| 3 | Specialized Built-Up | 0.5364 | 0.4278 | 0.4278 | 0.9923 | 0.4759 | 818189 |
| 4 | Specialized but Vegetative | 0.0626 | 0.4097 | 0.4097 | 0.9657 | 0.1086 | 222108 |
| 5 | Large Scale Networks | 0.2470 | 0.7331 | 0.7331 | 0.9761 | 0.3696 | 423239 |
| 6 | Non-urban / other | 0.9911 | 0.9091 | 0.9091 | 0.8918 | 0.9483 | 37161735 |
| **W-Avg** | | 0.9502 | 0.8910 | 0.8910 | 0.8977 | **0.9143** | |

Acc 0.8910 | Kappa 0.4672 | Mean F1 0.4667

### 2.3 Per-class F1 side-by-side (paper + ours)

| Class | Name | Paper | A4 last | Concat P4 | Concat P5 | MA P4 | MA P5 | S2 P4 | S1 P4 | S1 P5 |
|------:|------|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Dense Built-Up | 0.3639 | 0.4888 | 0.2565 | 0.4083 | 0.3522 | 0.3870 | 0.4119 | 0.2189 | 0.2203 |
| 2 | Sparse Built-Up | 0.6364 | 0.6649 | 0.4770 | 0.6896 | 0.6108 | 0.7179 | 0.6077 | 0.4329 | 0.4900 |
| 3 | Specialized Built-Up | 0.5894 | 0.4558 | 0.2137 | 0.6505 | 0.3679 | 0.6209 | 0.3607 | 0.1272 | 0.1436 |
| 4 | Specialized but Vegetative | 0.0584 | 0.0877 | 0.0919 | 0.1582 | 0.1794 | 0.1298 | 0.1301 | 0.0086 | 0.0074 |
| 5 | Large Scale Networks | 0.4064 | 0.3640 | 0.2472 | 0.4518 | 0.2806 | 0.4129 | 0.3068 | 0.1115 | 0.0956 |
| 6 | Non-urban / other | 0.9301 | 0.9339 | 0.9448 | 0.9653 | 0.9527 | 0.9621 | 0.9530 | 0.9610 | 0.9438 |

### 2.4 Per-class Precision side-by-side

| Class | Name | Paper | A4 last | Concat P4 | Concat P5 | MA P4 | MA P5 |
|------:|------|---:|---:|---:|---:|---:|---:|
| 1 | Dense Built-Up | 0.2308 | 0.4118 | 0.1688 | 0.5111 | 0.2732 | 0.4912 |
| 2 | Sparse Built-Up | 0.6260 | 0.5851 | 0.3552 | 0.5865 | 0.5155 | 0.6731 |
| 3 | Specialized Built-Up | 0.4794 | 0.3264 | 0.2013 | 0.6882 | 0.3742 | 0.5823 |
| 4 | Specialized but Vegetative | 0.0312 | 0.0484 | 0.0863 | 0.0925 | 0.1162 | 0.0750 |
| 5 | Large Scale Networks | 0.2736 | 0.2359 | 0.1524 | 0.3177 | 0.1698 | 0.2771 |
| 6 | Non-urban / other | 0.9965 | 0.9976 | 0.9877 | 0.9943 | 0.9909 | 0.9933 |

### 2.5 Per-class Recall side-by-side

| Class | Name | Paper | A4 last | Concat P4 | Concat P5 | MA P4 | MA P5 |
|------:|------|---:|---:|---:|---:|---:|---:|
| 1 | Dense Built-Up | 0.8599 | 0.6012 | 0.5336 | 0.3400 | 0.4956 | 0.3193 |
| 2 | Sparse Built-Up | 0.6472 | 0.7698 | 0.7260 | 0.8367 | 0.7493 | 0.7691 |
| 3 | Specialized Built-Up | 0.7647 | 0.7555 | 0.2277 | 0.6167 | 0.3618 | 0.6651 |
| 4 | Specialized but Vegetative | 0.4461 | 0.4680 | 0.0982 | 0.5477 | 0.3931 | 0.4812 |
| 5 | Large Scale Networks | 0.7898 | 0.7973 | 0.6532 | 0.7816 | 0.8079 | 0.8096 |
| 6 | Non-urban / other | 0.8719 | 0.8779 | 0.9055 | 0.9381 | 0.9174 | 0.9329 |

---

## 3. Ten-class - full globals + per-class

### 3.1 Global metrics

| Model | Acc | W-P | W-R | W-Sens | W-Spec | **W-F1** | Mean F1 | Mean Sens | Mean Spec | **Kappa** |
|-------|----:|----:|----:|-------:|-------:|---------:|--------:|----------:|----------:|----------:|
| **MA concat P5 full** (PARAM **104545**) | 0.8817 | 0.9040 | 0.8817 | 0.8817 | 0.9735 | **0.8885** | 0.6332 | 0.7294 | 0.9855 | **0.7950** |
| **Paper** ConvLSTM+Inception | - | 0.8977 | 0.8831 | 0.8831 | - | **0.8851** | - | - | - | **0.7945** |
| **S2-only P5 full** (PARAM **104890**) | 0.8819 | 0.9005 | 0.8819 | 0.8819 | 0.9716 | **0.8865** | 0.6230 | - | - | **0.7945** |
| **MA gated P5 full** (PARAM 104381) | 0.8789 | 0.8994 | 0.8789 | 0.8789 | 0.9728 | **0.8834** | - | - | - | **0.7895** |
| Concat U-TAE P5 full | 0.8740 | 0.8997 | 0.8740 | 0.8740 | 0.9659 | **0.8811** | 0.6043 | 0.6909 | 0.9840 | **0.7795** |
| A4 ConvLSTM | 0.8604 | 0.8947 | 0.8604 | 0.8604 | - | **0.8711** | 0.5853 | - | - | **0.7588** |
| **Paper** ConvLSTM-S2 | - | 0.9000 | 0.8517 | 0.8517 | - | **0.8696** | - | - | - | **0.7445** |
| **Paper** ConvLSTM-S1S2 | - | 0.8825 | 0.8482 | 0.8482 | - | **0.8600** | - | - | - | **0.7482** |
| MA concat P4 head | 0.8425 | 0.8790 | 0.8425 | 0.8425 | 0.9618 | **0.8547** | - | - | - | **0.7273** |
| S2-only P4 head | 0.8365 | 0.8643 | 0.8365 | 0.8365 | 0.9531 | **0.8437** | 0.5150 | 0.5972 | 0.9790 | **0.7152** |
| MA gated P4 head | 0.8272 | 0.8735 | 0.8272 | 0.8272 | 0.9707 | **0.8430** | 0.5111 | - | - | **0.7061** |
| **S1-only P5 full** (PARAM **104835**) | 0.8207 | 0.8607 | 0.8207 | 0.8207 | 0.9556 | **0.8365** | 0.5157 | - | - | **0.6939** |
| Concat U-TAE P4 head | 0.8262 | 0.8549 | 0.8262 | 0.8262 | 0.9512 | **0.8322** | 0.4591 | 0.5400 | 0.9777 | **0.6971** |
| **Paper** ConvLSTM-S1 | - | 0.8422 | 0.7836 | 0.7836 | - | **0.8055** | - | - | - | **0.6422** |
| S1-only P4 head | 0.7146 | 0.7792 | 0.7146 | 0.7146 | 0.9316 | **0.7342** | 0.3418 | 0.4168 | 0.9646 | **0.5396** |

### 3.2 Per-class Precision / Recall / Sens / Spec / F1

#### Paper Table 7 / 8 (RS 2023) — full transcription in [`PAPER_MODALITY_6CLASS.md`](PAPER_MODALITY_6CLASS.md)

| Model | W-P | W-R | **W-F1** | **Kappa** |
|-------|----:|----:|---------:|----------:|
| ConvLSTM-S1 | 0.8422 | 0.7836 | **0.8055** | **0.6422** |
| ConvLSTM-S2 | 0.9000 | 0.8517 | **0.8696** | **0.7445** |
| ConvLSTM-S1S2 | 0.8825 | 0.8482 | **0.8600** | **0.7482** |
| ConvLSTM+Inception-S1S2 | 0.8977 | 0.8831 | **0.8851** | **0.7945** |

##### Paper ConvLSTM-S1 (fair S1 baseline)

| Class | P | R | F1 |
|------:|--:|--:|---:|
| 1 | 0.1872 | 0.9247 | 0.3114 |
| 2 | 0.5718 | 0.5224 | 0.5460 |
| 3 | 0.4208 | 0.6480 | 0.5103 |
| 4 | 0.0892 | 0.4973 | 0.1512 |
| 5 | 0.2142 | 0.6183 | 0.3182 |
| 6 | 0.9649 | 0.8540 | 0.9060 |
| 7 | 0.8361 | 0.5625 | 0.6726 |
| 8 | 0.3890 | 0.4111 | 0.3997 |
| 9 | 0.7515 | 0.8280 | 0.7879 |
| 10 | 0.3143 | 0.4748 | 0.3782 |
| **W-Avg** | 0.8422 | 0.7836 | **0.8055** |

##### Paper ConvLSTM+Inception-S1S2 (main 10c row)

| Class | P | R | F1 |
|------:|--:|--:|---:|
| 1 | 0.3870 | 0.7190 | 0.5031 |
| 2 | 0.6672 | 0.8066 | 0.7303 |
| 3 | 0.4612 | 0.7632 | 0.5749 |
| 4 | 0.1863 | 0.3643 | 0.2465 |
| 5 | 0.4290 | 0.7560 | 0.5474 |
| 6 | 0.9718 | 0.9558 | 0.9637 |
| 7 | 0.8869 | 0.8512 | 0.8687 |
| 8 | 0.7422 | 0.3949 | 0.5155 |
| 9 | 0.8585 | 0.8643 | 0.8614 |
| 10 | 0.4654 | 0.7074 | 0.5614 |
| **W-Avg** | 0.8977 | 0.8831 | **0.8851** |

#### S2-only P5 test (PARAM **104890**) — on laptop

| Class | Precision | Recall | Sens | Spec | F1 |
|------:|----------:|-------:|-----:|-----:|---:|
| 1 | 0.4164 | 0.4950 | 0.4950 | 0.9976 | 0.4523 |
| 2 | 0.6646 | 0.8068 | 0.8068 | 0.9872 | 0.7289 |
| 3 | 0.5227 | 0.7836 | 0.7836 | 0.9850 | 0.6271 |
| 4 | 0.2377 | 0.3297 | 0.3297 | 0.9941 | 0.2762 |
| 5 | 0.3409 | 0.7783 | 0.7783 | 0.9839 | 0.4742 |
| 6 | 0.9792 | 0.9503 | 0.9503 | 0.9650 | 0.9645 |
| 7 | 0.8259 | 0.8380 | 0.8380 | 0.9891 | 0.8319 |
| 8 | 0.7125 | 0.4323 | 0.4323 | 0.9885 | 0.5381 |
| 9 | 0.8829 | 0.8667 | 0.8667 | 0.9769 | 0.8747 |
| 10 | 0.3262 | 0.7929 | 0.7929 | 0.9861 | 0.4622 |
| **W-Avg** | 0.9005 | 0.8819 | 0.8819 | 0.9716 | **0.8865** |

Acc 0.8819 | Kappa **0.7945** | Mean F1 0.6230 · `results/concat_utae/run_c10_s2_full_v0/`

#### S1-only P5 test (PARAM **104835**) — on laptop

| Class | Precision | Recall | Sens | Spec | F1 |
|------:|----------:|-------:|-----:|-----:|---:|
| 1 | 0.2186 | 0.8655 | 0.8655 | 0.9895 | 0.3490 |
| 2 | 0.6076 | 0.6166 | 0.6166 | 0.9875 | 0.6121 |
| 3 | 0.5039 | 0.5004 | 0.5004 | 0.9897 | 0.5022 |
| 4 | 0.1413 | 0.3432 | 0.3432 | 0.9883 | 0.2002 |
| 5 | 0.1505 | 0.6319 | 0.6319 | 0.9618 | 0.2431 |
| 6 | 0.9665 | 0.9088 | 0.9088 | 0.9453 | 0.9368 |
| 7 | 0.8185 | 0.6342 | 0.6342 | 0.9913 | 0.7146 |
| 8 | 0.4820 | 0.3995 | 0.3995 | 0.9718 | 0.4369 |
| 9 | 0.8158 | 0.8273 | 0.8273 | 0.9625 | 0.8215 |
| 10 | 0.2635 | 0.4823 | 0.4823 | 0.9885 | 0.3408 |
| **W-Avg** | 0.8607 | 0.8207 | 0.8207 | 0.9556 | **0.8365** |

Acc 0.8207 | Kappa **0.6939** | Mean F1 0.5157 · `results/concat_utae/run_c10_s1_full_v0/`

#### A4 ConvLSTM (no Spec in JSON)

| Class | Name | Precision | Recall | Sensitivity | Specificity | F1 | Support |
|------:|------|----------:|-------:|------------:|------------:|---:|--------:|
| 1 | Dense Built-Up | 0.3207 | 0.7085 | 0.7085 | - | 0.4415 | 135479 |
| 2 | Sparse Built-Up | 0.6831 | 0.6906 | 0.6906 | - | 0.6868 | 1216210 |
| 3 | Specialized Built-Up | 0.3699 | 0.8015 | 0.8015 | - | 0.5062 | 818189 |
| 4 | Specialized but Vegetative | 0.1187 | 0.3064 | 0.3064 | - | 0.1712 | 222108 |
| 5 | Large Scale Networks | 0.3411 | 0.7528 | 0.7528 | - | 0.4694 | 423239 |
| 6 | Arable land | 0.9770 | 0.9431 | 0.9431 | - | 0.9597 | 25350769 |
| 7 | Permanent crops | 0.8185 | 0.8255 | 0.8255 | - | 0.8220 | 2332261 |
| 8 | Grassland | 0.6724 | 0.4245 | 0.4245 | - | 0.5204 | 2463353 |
| 9 | Forest | 0.8967 | 0.7880 | 0.7880 | - | 0.8388 | 6678255 |
| 10 | Water | 0.2984 | 0.8148 | 0.8148 | - | 0.4368 | 337097 |
| **W-Avg** | | 0.8947 | 0.8604 | 0.8604 | - | **0.8711** | |

Acc 0.8604 | Kappa 0.7588 | Mean F1 0.5853

#### Concat U-TAE P4 head

| Class | Name | Precision | Recall | Sensitivity | Specificity | F1 | Support |
|------:|------|----------:|-------:|------------:|------------:|---:|--------:|
| 1 | Dense Built-Up | 0.2229 | 0.5141 | 0.5141 | 0.9939 | 0.3110 | 135479 |
| 2 | Sparse Built-Up | 0.4696 | 0.7249 | 0.7249 | 0.9743 | 0.5700 | 1216210 |
| 3 | Specialized Built-Up | 0.5981 | 0.0930 | 0.0930 | 0.9987 | 0.1610 | 818189 |
| 4 | Specialized but Vegetative | 0.0401 | 0.1378 | 0.1378 | 0.9816 | 0.0621 | 222108 |
| 5 | Large Scale Networks | 0.2243 | 0.6240 | 0.6240 | 0.9769 | 0.3300 | 423239 |
| 6 | Arable land | 0.9623 | 0.9357 | 0.9357 | 0.9364 | 0.9488 | 25350769 |
| 7 | Permanent crops | 0.5624 | 0.4624 | 0.4624 | 0.9777 | 0.5076 | 2332261 |
| 8 | Grassland | 0.5910 | 0.3894 | 0.3894 | 0.9823 | 0.4695 | 2463353 |
| 9 | Forest | 0.8581 | 0.8575 | 0.8575 | 0.9716 | 0.8578 | 6678255 |
| 10 | Water | 0.2603 | 0.6612 | 0.6612 | 0.9840 | 0.3736 | 337097 |
| **W-Avg** | | 0.8549 | 0.8262 | 0.8262 | 0.9512 | **0.8322** | |

Acc 0.8262 | Kappa 0.6971 | Mean F1 0.4591

#### Concat U-TAE P5 full

| Class | Name | Precision | Recall | Sensitivity | Specificity | F1 | Support |
|------:|------|----------:|-------:|------------:|------------:|---:|--------:|
| 1 | Dense Built-Up | 0.4260 | 0.5455 | 0.5455 | 0.9975 | 0.4784 | 135479 |
| 2 | Sparse Built-Up | 0.6422 | 0.8215 | 0.8215 | 0.9856 | 0.7209 | 1216210 |
| 3 | Specialized Built-Up | 0.4656 | 0.7708 | 0.7708 | 0.9815 | 0.5806 | 818189 |
| 4 | Specialized but Vegetative | 0.1076 | 0.2997 | 0.2997 | 0.9861 | 0.1584 | 222108 |
| 5 | Large Scale Networks | 0.3652 | 0.7606 | 0.7606 | 0.9859 | 0.4935 | 423239 |
| 6 | Arable land | 0.9736 | 0.9552 | 0.9552 | 0.9552 | 0.9644 | 25350769 |
| 7 | Permanent crops | 0.8973 | 0.6831 | 0.6831 | 0.9952 | 0.7757 | 2332261 |
| 8 | Grassland | 0.7088 | 0.4443 | 0.4443 | 0.9880 | 0.5462 | 2463353 |
| 9 | Forest | 0.8894 | 0.8508 | 0.8508 | 0.9788 | 0.8697 | 6678255 |
| 10 | Water | 0.3222 | 0.7778 | 0.7778 | 0.9861 | 0.4557 | 337097 |
| **W-Avg** | | 0.8997 | 0.8740 | 0.8740 | 0.9659 | **0.8811** | |

Acc 0.8740 | Kappa 0.7795 | Mean F1 0.6043

### 3.3 Per-class F1 side-by-side

| Class | Name | A4 | Concat P4 | Concat P5 |
|------:|------|---:|---:|---:|
| 1 | Dense Built-Up | 0.4415 | 0.3110 | 0.4784 |
| 2 | Sparse Built-Up | 0.6868 | 0.5700 | 0.7209 |
| 3 | Specialized Built-Up | 0.5062 | 0.1610 | 0.5806 |
| 4 | Specialized but Vegetative | 0.1712 | 0.0621 | 0.1584 |
| 5 | Large Scale Networks | 0.4694 | 0.3300 | 0.4935 |
| 6 | Arable land | 0.9597 | 0.9488 | 0.9644 |
| 7 | Permanent crops | 0.8220 | 0.5076 | 0.7757 |
| 8 | Grassland | 0.5204 | 0.4695 | 0.5462 |
| 9 | Forest | 0.8388 | 0.8578 | 0.8697 |
| 10 | Water | 0.4368 | 0.3736 | 0.4557 |

### 3.4 Per-class Precision side-by-side

| Class | Name | A4 | Concat P4 | Concat P5 |
|------:|------|---:|---:|---:|
| 1 | Dense Built-Up | 0.3207 | 0.2229 | 0.4260 |
| 2 | Sparse Built-Up | 0.6831 | 0.4696 | 0.6422 |
| 3 | Specialized Built-Up | 0.3699 | 0.5981 | 0.4656 |
| 4 | Specialized but Vegetative | 0.1187 | 0.0401 | 0.1076 |
| 5 | Large Scale Networks | 0.3411 | 0.2243 | 0.3652 |
| 6 | Arable land | 0.9770 | 0.9623 | 0.9736 |
| 7 | Permanent crops | 0.8185 | 0.5624 | 0.8973 |
| 8 | Grassland | 0.6724 | 0.5910 | 0.7088 |
| 9 | Forest | 0.8967 | 0.8581 | 0.8894 |
| 10 | Water | 0.2984 | 0.2603 | 0.3222 |

### 3.5 Per-class Recall side-by-side

| Class | Name | A4 | Concat P4 | Concat P5 |
|------:|------|---:|---:|---:|
| 1 | Dense Built-Up | 0.7085 | 0.5141 | 0.5455 |
| 2 | Sparse Built-Up | 0.6906 | 0.7249 | 0.8215 |
| 3 | Specialized Built-Up | 0.8015 | 0.0930 | 0.7708 |
| 4 | Specialized but Vegetative | 0.3064 | 0.1378 | 0.2997 |
| 5 | Large Scale Networks | 0.7528 | 0.6240 | 0.7606 |
| 6 | Arable land | 0.9431 | 0.9357 | 0.9552 |
| 7 | Permanent crops | 0.8255 | 0.4624 | 0.6831 |
| 8 | Grassland | 0.4245 | 0.3894 | 0.4443 |
| 9 | Forest | 0.7880 | 0.8575 | 0.8508 |
| 10 | Water | 0.8148 | 0.6612 | 0.7778 |

---

## 4. P3 probes (val only - not paper-facing)

| Setting | Best layer | Val W-F1 | Artifacts |
|---------|------------|---------:|-----------|
| 6c linear probes | L2 | **0.7477** | `results/concat_utae/probe_c6_v0/` |
| 10c linear probes | L1 | **0.8009** | `results/concat_utae/probe_c10_v0/` |

---

## 5. Honest notes

- **Fair modality match:** S1-only vs paper **ConvLSTM-S1**, S2-only vs **ConvLSTM-S2** (not vs Inception-S1S2). Full 6c+10c tables: [`PAPER_MODALITY_6CLASS.md`](PAPER_MODALITY_6CLASS.md).
- **6c S1 vs paper ConvLSTM-S1 (Table 6 κ=0.3929):** P4 = +0.011 W-F1 / +0.0045 κ; P5 = −0.003 W-F1 / −0.039 κ.
- **6c S2 vs paper ConvLSTM-S2 (Table 6 κ=0.4223):** P4 +0.021 / +0.0435; P5 +0.024 / +0.059.
- **10c S1 vs paper ConvLSTM-S1 (0.8055 / 0.6422):** P4 test **below** (−0.071 / −0.103). **P5 test** (**104835**) **above** — W-F1 **0.8365** / κ **0.6939** → **+0.031 / +0.052**.
- **10c S2 vs paper ConvLSTM-S2 (0.8696 / 0.7445):** P4 test below (−0.026 / −0.029). **P5 test** (**104890**) **above** — W-F1 **0.8865** / κ **0.7945** → **+0.017 / +0.050**. (κ matches paper Inc by coincidence — do not claim S2 beats fusion.)
- **6c both:** Concat U-TAE **P5** still leads. MA gated P5 is close. MA concat-fuse peaks at **P4** (0.9218); P5 (0.9143) is lower.
- **10c Task M:** **MA concat P5** W-F1 **0.8885** / κ **0.7950** **beats paper Inc** (0.8851 / 0.7945) and gated P5 (0.8834 / 0.7895). On 10c, concat fuse > gated (opposite of 6c, where gated led among MA).
- **Task H:** **deferred** (future work). Not required for Task M write-up.
- Always quote **per-class F1 (esp. 1, 2, 4)** beside W-F1; W-F1 is majority-dominated (class 6 / arable).
- A4 6c **report** = `last.pt` ep25 (W-F1 0.9037), not `best.pt` (0.9098) - see `TABLE5_TEST_FOR_SIR.md`.
- S1/S2 **paper-facing schedule** = **P5 test**; still show P4 when P5 regresses (S1).

---

## 6. Checklist - remaining

- [x] S1 P5 re-test metrics confirmed (102355; ignore exit 9 if JSON written)
- [x] MA concat 6c **P4+P5 test** on laptop (P4 0.9218 / P5 0.9143)
- [x] S2-only 6c **P5 test** on laptop (102628; W-F1 0.9199 / κ 0.4809)
- [x] MA gated **10c** P4 + P5 **test** (PARAM; sync JSON to laptop)
- [x] MA concat / S1 / S2 **10c P4 test** (PARAM headlines logged)
- [x] Sync S1/S2 **10c P5** JSON + plots to laptop (`msge_10c_p5.tgz`)
- [x] Sync **MA concat 10c P5** to laptop (`msge_ma_c10_concat_p5.tgz`)
- [x] MA concat **10c P5** train+test (**104173** / **104545**; W-F1 **0.8885** / κ **0.7950**)
- [x] S1 / S2 **10c P5** train+test (**104546**/**104658** train; **104835**/**104890** test)
- [x] Paste **paper Table 7/8** into section 3 + [`PAPER_MODALITY_6CLASS.md`](PAPER_MODALITY_6CLASS.md)
- [ ] Task H on best backbone → **deferred / future work** (not needed for Task M write-up)
- [ ] Re-run board fill from JSONs after sync

---

## 7. Download from PARAM (metrics + training graphs)

### Still on PARAM only (need laptop sync)

| Path on PARAM | Why |
|---------------|-----|
| `results/concat_utae/run_c10_s1_full_v0/` | S1 P5 **test** JSON+md (**104835**) |
| `results/concat_utae/run_c10_s2_full_v0/` | S2 P5 **test** JSON+md (**104890**) |
| `checkpoints/run_c10_s1_full_v0/{history.json,history_plot.png,best_metrics.json}` | S1 P5 train curves |
| `checkpoints/run_c10_s2_full_v0/{history.json,history_plot.png,best_metrics.json}` | S2 P5 train curves |
| `results/ma_utae/ma_c10_concat_full_v0/` (if exists) | MA concat 10c P5 |
| `checkpoints/ma_c10_concat_full_v0/` (if exists) | MA concat 10c P5 train |

### Fast path (laptop PowerShell)

```powershell
cd E:\MTP\earth2

# 1) S1/S2 P5 test metrics (required now)
scp -r rihak_iitp@paramrudra.iitp.ac.in:~/MTP/earth2/multisenge_utae/results/concat_utae/run_c10_s1_full_v0 ./multisenge_utae/results/concat_utae/
scp -r rihak_iitp@paramrudra.iitp.ac.in:~/MTP/earth2/multisenge_utae/results/concat_utae/run_c10_s2_full_v0 ./multisenge_utae/results/concat_utae/

# 2) S1/S2 P5 train plots into same result folders
New-Item -ItemType Directory -Force -Path multisenge_utae\results\concat_utae\run_c10_s1_full_v0 | Out-Null
New-Item -ItemType Directory -Force -Path multisenge_utae\results\concat_utae\run_c10_s2_full_v0 | Out-Null
scp rihak_iitp@paramrudra.iitp.ac.in:~/MTP/earth2/multisenge_utae/checkpoints/run_c10_s1_full_v0/history_plot.png ./multisenge_utae/results/concat_utae/run_c10_s1_full_v0/
scp rihak_iitp@paramrudra.iitp.ac.in:~/MTP/earth2/multisenge_utae/checkpoints/run_c10_s1_full_v0/history.json ./multisenge_utae/results/concat_utae/run_c10_s1_full_v0/
scp rihak_iitp@paramrudra.iitp.ac.in:~/MTP/earth2/multisenge_utae/checkpoints/run_c10_s1_full_v0/best_metrics.json ./multisenge_utae/results/concat_utae/run_c10_s1_full_v0/
scp rihak_iitp@paramrudra.iitp.ac.in:~/MTP/earth2/multisenge_utae/checkpoints/run_c10_s2_full_v0/history_plot.png ./multisenge_utae/results/concat_utae/run_c10_s2_full_v0/
scp rihak_iitp@paramrudra.iitp.ac.in:~/MTP/earth2/multisenge_utae/checkpoints/run_c10_s2_full_v0/history.json ./multisenge_utae/results/concat_utae/run_c10_s2_full_v0/
scp rihak_iitp@paramrudra.iitp.ac.in:~/MTP/earth2/multisenge_utae/checkpoints/run_c10_s2_full_v0/best_metrics.json ./multisenge_utae/results/concat_utae/run_c10_s2_full_v0/

# 3) Optional: older 10c pack + MA concat P5 if present
powershell -ExecutionPolicy Bypass -File .\multisenge_utae\sync_10c_from_param.ps1
scp -r rihak_iitp@paramrudra.iitp.ac.in:~/MTP/earth2/multisenge_utae/results/ma_utae/ma_c10_concat_full_v0 ./multisenge_utae/results/ma_utae/ 2>$null
```

### PARAM one-shot pack then pull

```bash
# on PARAM
cd ~/MTP/earth2
ls -la multisenge_utae/results/ma_utae/ma_c10_concat_full_v0 2>/dev/null || echo "no ma concat P5 yet"
tar czf /tmp/msge_10c_p5.tgz \
  multisenge_utae/results/concat_utae/run_c10_s1_full_v0 \
  multisenge_utae/results/concat_utae/run_c10_s2_full_v0 \
  multisenge_utae/checkpoints/run_c10_s1_full_v0/history_plot.png \
  multisenge_utae/checkpoints/run_c10_s1_full_v0/history.json \
  multisenge_utae/checkpoints/run_c10_s1_full_v0/best_metrics.json \
  multisenge_utae/checkpoints/run_c10_s2_full_v0/history_plot.png \
  multisenge_utae/checkpoints/run_c10_s2_full_v0/history.json \
  multisenge_utae/checkpoints/run_c10_s2_full_v0/best_metrics.json \
  multisenge_utae/results/ma_utae/ma_c10_concat_full_v0 \
  multisenge_utae/checkpoints/ma_c10_concat_full_v0/history_plot.png \
  multisenge_utae/checkpoints/ma_c10_concat_full_v0/history.json \
  2>/dev/null
ls -lh /tmp/msge_10c_p5.tgz
```

```powershell
# on laptop
cd E:\MTP\earth2
scp rihak_iitp@paramrudra.iitp.ac.in:/tmp/msge_10c_p5.tgz .
tar xzf msge_10c_p5.tgz
# copy plots into results folders if they landed under checkpoints/
Copy-Item multisenge_utae\checkpoints\run_c10_s1_full_v0\history* multisenge_utae\results\concat_utae\run_c10_s1_full_v0\ -ErrorAction SilentlyContinue
Copy-Item multisenge_utae\checkpoints\run_c10_s2_full_v0\history* multisenge_utae\results\concat_utae\run_c10_s2_full_v0\ -ErrorAction SilentlyContinue
```

### PARAM check before download

```bash
cd ~/MTP/earth2
ls -la multisenge_utae/results/concat_utae/run_c10_s{1,2}_full_v0/
ls -la multisenge_utae/checkpoints/run_c10_s{1,2}_full_v0/history_plot.png
ls -la multisenge_utae/results/ma_utae/ma_c10_*/ 2>/dev/null
squeue -u $USER
```
