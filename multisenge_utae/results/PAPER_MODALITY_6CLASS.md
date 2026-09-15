# Paper modality ablations (RS 2023) — 6-class test 31UEQ

Transcribed from Wenger et al. *Remote Sensing* 2023 (ConvLSTM-S1 / S2 / S1S2 / +Inception).
Use these rows when comparing **our** S1-only / S2-only / both U-TAE — **not** S1-only vs paper Inception-S1S2.

**Last updated:** 2026-09-14

## Kappa (paper Table 6)

| Model | Kappa |
|-------|------:|
| ConvLSTM-S1 | **0.3929** |
| ConvLSTM-S2 | **0.4223** (paper best in this table) |
| ConvLSTM-S1S2 | 0.3852 |
| ConvLSTM+Inception-S1S2 | 0.4186 |

## Per-class + W-Avg (paper)

### ConvLSTM-S1

| Class | P | R | F1 |
|------:|--:|--:|---:|
| 1 | 0.1335 | 0.9397 | 0.2337 |
| 2 | 0.4476 | 0.3809 | 0.4116 |
| 3 | 0.3560 | 0.5813 | 0.4416 |
| 4 | 0.0775 | 0.5072 | 0.1344 |
| 5 | 0.1313 | 0.5516 | 0.2122 |
| 6 | 0.9937 | 0.8937 | 0.9410 |
| **W-Avg** | 0.9469 | 0.8661 | **0.9001** |

### ConvLSTM-S2

| Class | P | R | F1 |
|------:|--:|--:|---:|
| 1 | 0.2579 | 0.8704 | 0.3980 |
| 2 | 0.5575 | 0.7268 | 0.6310 |
| 3 | 0.3100 | 0.7763 | 0.4431 |
| 4 | 0.0528 | 0.4858 | 0.0953 |
| 5 | 0.2137 | 0.7995 | 0.3372 |
| 6 | 0.9979 | 0.8663 | 0.9274 |
| **W-Avg** | 0.9544 | 0.8574 | **0.8958** |

### ConvLSTM-S1S2

| Class | P | R | F1 |
|------:|--:|--:|---:|
| 1 | 0.3122 | 0.7624 | 0.4430 |
| 2 | 0.5671 | 0.7706 | 0.6533 |
| 3 | 0.4654 | 0.6859 | 0.5545 |
| 4 | 0.0314 | 0.5739 | 0.0595 |
| 5 | 0.2745 | 0.8085 | 0.4099 |
| 6 | 0.9971 | 0.8446 | 0.9145 |
| **W-Avg** | 0.9578 | 0.8369 | **0.8875** |

### ConvLSTM+Inception-S1S2 (main paper row)

| Class | P | R | F1 |
|------:|--:|--:|---:|
| 1 | 0.2308 | 0.8599 | 0.3639 |
| 2 | 0.6260 | 0.6472 | 0.6364 |
| 3 | 0.4794 | 0.7647 | 0.5894 |
| 4 | 0.0312 | 0.4461 | 0.0584 |
| 5 | 0.2736 | 0.7898 | 0.4064 |
| 6 | 0.9965 | 0.8719 | 0.9301 |
| **W-Avg** | 0.9591 | 0.8596 | **0.9018** |

## Ours vs paper (fair modality match)

Report deltas **plainly** (same wording for wins and losses). Small W-F1 gaps (~0.01) are small either way; **kappa** can tell a different story.

| Setting | Paper model | Paper W-F1 | Paper κ | Ours | Ours W-F1 | Ours κ | Δ W-F1 | Δ κ |
|---------|-------------|------------|---------|------|-----------|--------|--------|-----|
| S1-only | ConvLSTM-S1 | 0.9001 | **0.3929** | U-TAE P4 | 0.9111 | 0.3974 | **+0.0110** | **+0.0045** |
| S1-only | ConvLSTM-S1 | 0.9001 | **0.3929** | U-TAE **P5** (schedule) | 0.8970 | 0.3537 | **−0.0031** | **−0.0392** |
| S2-only | ConvLSTM-S2 | 0.8958 | **0.4223** | U-TAE P4 | 0.9171 | 0.4658 | **+0.0213** | **+0.0435** |
| S2-only | ConvLSTM-S2 | 0.8958 | **0.4223** | U-TAE P5 | — | — | TBD | TBD |
| Both | ConvLSTM+Inc | 0.9018 | 0.4186 | Concat U-TAE P5 | 0.9387 | 0.5757 | **+0.0369** | **+0.1571** |
| Both | ConvLSTM+Inc | 0.9018 | 0.4186 | A4 (our Inc) | 0.9037 | 0.4424 | **+0.0019** | **+0.0238** |

### S1-only — neutral reading (no soft language)

Paper **Table 6** kappa for ConvLSTM-S1 = **0.3929** (use this, not any other figure).

| Ours | vs paper W-F1 | vs paper κ (0.3929) | One-line |
|------|---------------|---------------------|----------|
| P4 | +0.011 | **+0.0045** | Slightly higher W-F1 and slightly higher kappa |
| P5 | −0.003 | **−0.0392** | Slightly lower W-F1; clearly lower kappa |

Prefer: “P4 +0.011 W-F1 / +0.0045 κ; P5 −0.003 W-F1 / −0.039 κ.” Do not call small W-F1 gaps “win” vs “tie.”

**Do not** compare S1-only U-TAE to ConvLSTM+Inception-S1S2 (different inputs).
