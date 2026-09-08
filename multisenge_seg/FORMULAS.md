# Phase 2 / Pillar A - Formulas (loss, activations, metrics)

**Scope:** MultiSenGE pixel segmentation (A4 ConvLSTM+Inception + A5 U-TAE).  
**Code:** `multisenge_seg/metrics.py`, `multisenge_seg/train.py`, `multisenge_utae/train.py`, `multisenge_utae/models/{utae,ltae}.py`, `multisenge_utae/probe_layers.py`.  
**Why this file:** Phase 2 report uses many metrics; this is the single formula note for CNN / U-TAE.  
**Not this file:** `writeup/METRICS_FORMULAS.md` is **LULCDial VLM** only (S1 dB + 70/30 hash split).

**Indexing:** Ground-reference classes in paper are **1..C**. In code / CM they are stored as **0..C-1**. Nodata / ignore = **255** (dropped from loss and metrics).

---

## 1. Output activation - Softmax

For a pixel with logits `z = (z_0, …, z_{C-1})`:

```text
p_c = exp(z_c) / sum_k exp(z_k)     for c = 0..C-1
```

| Piece | Meaning |
|-------|---------|
| `z_c` | Logit for class `c` from the final conv head (`out_conv`) |
| `p_c` | Class probability (sums to 1 over `c`) |

**Where used:**
- Training: `nn.CrossEntropyLoss` applies **log-softmax + NLL** on logits (you do **not** Softmax before CE).
- Reporting probs / analysis: Softmax on logits.
- L-TAE: Softmax over **time** (dates), not over classes (see §3.2).

**Why:** Multi-class exclusive labels per pixel (one land-cover class). Not Sigmoid (that would be multi-label).

---

## 2. Network nonlinearities (U-TAE and A4)

| Module | Activation / norm | Why in our runs |
|--------|-------------------|-----------------|
| U-TAE `ConvBlock` / ups | **ReLU** after conv | Stock U-TAE (Garnot & Landrieu) |
| U-TAE spatial path | **GroupNorm** | Stable with small batch (**B = 2**) |
| L-TAE input/output | **GroupNorm** | Same |
| L-TAE MLP after attention | **Linear → BatchNorm1d → ReLU**, Dropout 0.2 | Stock L-TAE |
| L-TAE attention | **Softmax over T** | Temporal weights |
| A4 ConvLSTM + VGG-style U-Net | ReLU CNN stack | Match RS-2023 replicate |

---

## 3. U-TAE formulas (architecture + L-TAE)

**Code:** `multisenge_utae/models/utae.py`, `ltae.py`.  
**Input:** `x` with shape `B × T × C × H × W`, `T=4`, `C=12` (ch **0-9 = S2**, **10-11 = S1 VV/VH**), `H=W=256`.  
Built by `stack_modalities()` in `multisenge_utae/data.py`.

### 3.1 Forward path

```text
L0  = in_conv(x)                 # B×T×64×256×256
L1  = down_blocks[0](L0)         # B×T×64×128×128
L2  = down_blocks[1](L1)         # B×T×64×64×64
Lb  = down_blocks[2](L2)         # B×T×128×32×32   (bottleneck, still has time)

(f, α) = LTAE2d(Lb, positions)   # f: B×128×32×32 (time collapsed)
                                 # α: n_head × B × T × 32 × 32
                                 # Probe level L3 = f

for each skip in {L2, L1, L0}:
    skip_agg = TemporalAggregator(skip, α)   # weight dates by α, collapse T
    f = UpConv(f, skip_agg)

logits = out_conv(f)             # B×C×256×256   (C = num_classes)
# train: CrossEntropyLoss(logits, mask)      # Softmax inside CE
# infer: pred = argmax_c(logits)
```

| Name in probes | Tensor |
|----------------|--------|
| L0 | after `in_conv` (still has `T`) |
| L1 | after first down |
| L2 | after second down |
| L3 | after L-TAE (`f`, no `T`) |

**Why this structure:** Official U-TAE: shared spatial encoder over time → **one** L-TAE at 1/8 res → U-Net decoder with attention-weighted skips. Same **wCE** as A4 (§4-§5) for a fair bake-off.

### 3.2 L-TAE temporal Softmax

Per head / location, attention over `T` dates:

```text
scores_t = (queries, keys)_t / sqrt(d_k)     # scaled dot-product (stock L-TAE)
α_t      = Softmax_t(scores_t)               # sum_t α_t = 1
out      = sum_t α_t · value_t
```

Default L-TAE: `n_head=16`, `d_k=4`, `d_model=256`, MLP `[256, 128]`.  
`positions` = month indices `(7, 8, 9, 11)` via positional encoding.

**Attention entropy** (analysis / future Priority 3 only; **not** training loss):

```text
H(α) = - sum_t α_t · log(α_t)
```

| | Meaning |
|--|---------|
| High `H` | Attention spread across many dates |
| Low `H` | Attention peaks on one / few dates |

**Why Softmax over time:** Turns date scores into normalized temporal weights for the 4 MultiSenGE dates.

### 3.3 P4 head vs P5 full (no new loss)

From `UTAE.set_train_mode`:

| Mode | Trainable | Frozen | Loss |
|------|-----------|--------|------|
| **P4 `head`** | `up_blocks` + `out_conv` | `in_conv`, `down_blocks`, `temporal_encoder` (**L-TAE**), `temporal_aggregator` | wCE |
| **P5 `full`** | all parameters | none | wCE (init from P4 `best.pt`) |

---

## 4. Training loss - Cross-Entropy (CE)

True class `y`, predicted probs `p` (from Softmax of logits):

```text
CE = - log(p_y)
```

Implemented as `nn.CrossEntropyLoss(..., ignore_index=255)` on **logits** (batch mean over valid pixels).

**Why:** Standard exclusive multi-class segmentation; matches Wenger et al. RS 2023 setup.

---

## 5. Weighted Cross-Entropy (Weighted CE / wCE)

```text
L_wCE = - w_y · log(p_y)
```

Class weights (`class_weights_from_counts` in `metrics.py`):

```text
count_c = # train pixels of class c   (floored at 1)
inv_c   = 1 / count_c
w_c     = inv_c / sum_k(inv_k) · C     # renormalize so weights sum to C
```

| Piece | Meaning |
|-------|---------|
| Rare class | Large `w_c` |
| Majority (e.g. arable / “other”) | Small `w_c` |

**Why we used it:**
1. Paper-style handling of MultiSenGE imbalance (urban << non-urban).
2. Without it, models collapse to majority; W-F1 can look fine while UF fails.
3. **Same weights recipe for A4 and U-TAE** so comparison is fair.

**What it does not do:** Target a specific confuse pair (Dense↔Sparse). That is Gap B / Priority 2.

---

## 6. P3 probe - Multinomial logistic regression

**Code:** `probe_layers.py` → `sklearn.linear_model.LogisticRegression`  
`multi_class="multinomial"`, `class_weight="balanced"`, `penalty=l2` (default), solver `saga` if n_pixels > 100000 else `lbfgs`.

**Role:** U-TAE weights **frozen**. Probe only asks: are layer features linearly separable?

### 6.1 Feature vector per pixel

```text
x ∈ R^d
  L0, L1, L2: mean over time of feature map → d = #channels (64)
  L3:         post L-TAE map               → d = 128
```

Pixels subsampled from train / val patches (`pixels_per_patch`, default 512).

### 6.2 Multinomial (Softmax) logistic regression

Per class `c`, weights `β_c ∈ R^d`, bias `b_c`:

```text
z_c = β_c · x + b_c
p_c = exp(z_c) / sum_k exp(z_k)
ŷ   = argmax_c p_c
```

**Objective** (conceptual; sklearn fits this):

```text
L_LR = - sum_n w_{y_n} · log(p_{y_n}(x_n)) + (1/(2C_reg)) · ||β||^2
```

(`C_reg` is sklearn’s `C`, default 1.0; larger `C` = less L2.)

| Setting | Meaning |
|---------|---------|
| `multinomial` | One Softmax over all classes (not OvR binary LRs) |
| `class_weight=balanced` | `w_c ∝ n_samples / (C · count_c)` - rare classes upweighted |
| `saga` / `lbfgs` | Optimizers for large / smaller pixel counts |

**Why multinomial LR (not the U-TAE head):**
1. Weak **linear** readout → scores representation quality of L0-L3.
2. Same multi-class Softmax assumption as the segmentation head.
3. Breast-style P3; metrics afterward use the **same** CM formulas (§7-§12) on **val** pixels.
4. RF exists in code (`--probe rf`); **reported** 6c/10c tables used **linear / multinomial LR**.

**Split:** fit on **train** tiles → evaluate on **val** (31UFP+31UGP). Probes are **not** scored on test 31UEQ.

---

## 7. Confusion matrix

Rows = GT, columns = prediction (classes `0..C-1`):

```text
CM[i, j] = # pixels with GT = i and Pred = j
```

One-vs-rest for class `i`:

```text
TP_i = CM[i, i]
FP_i = sum_r CM[r, i] - TP_i
FN_i = sum_c CM[i, c] - TP_i
TN_i = N - TP_i - FP_i - FN_i     # N = sum(CM)
```

**Why:** Single source for P/R/Spec/F1/Kappa; paper Table 5 style; stored as `confusion_matrix` in JSON.

---

## 8. Per-class Precision, Recall, Sensitivity, Specificity, F1

```text
Precision_i    = TP_i / (TP_i + FP_i)       # 0 if denom 0
Recall_i       = TP_i / (TP_i + FN_i)       # = Sensitivity_i
Sensitivity_i  = Recall_i                  # breast-TL name
Specificity_i  = TN_i / (TN_i + FP_i)
F1_i           = 2·TP_i / (2·TP_i + FP_i + FN_i)
```

| Metric | Why we report it |
|--------|------------------|
| Precision | Of predicted class i, how many correct? |
| Recall / Sensitivity | Of true class i, how many found? (paper Table 5) |
| Specificity | One-vs-rest true-negative rate (breast-style) |
| F1 | Balance Prec and Rec; paper per-class headline |

---

## 9. Support-weighted averages (paper "W-Avg")

`support_i = sum_j CM[i, j]` (GT count for class i):

```text
W-Precision   = sum_i (Precision_i · support_i) / sum_i support_i
W-Recall      = sum_i (Recall_i · support_i) / sum_i support_i      # = W-Sensitivity
W-Specificity = sum_i (Specificity_i · support_i) / sum_i support_i
W-F1          = sum_i (F1_i · support_i) / sum_i support_i
```

**Why W-F1 headline:** Matches Wenger et al. RS 2023 Table 5 W-Avg.  
**Caveat:** Dominated by majority classes; always also report per-class F1 and Kappa.

---

## 10. Overall accuracy

```text
Accuracy = trace(CM) / sum(CM)
```

**Why:** Simple; weak under imbalance; still logged.

---

## 11. Mean F1 (unweighted)

```text
MeanF1 = (1/C) · sum_i F1_i
```

**Why:** Equal weight per class. Early-stopping / LR monitor uses **W-F1**, not MeanF1 (paper-facing).

---

## 12. Cohen's Kappa

```text
P_o = Accuracy
P_e = sum_i (row_sum_i · col_sum_i) / N^2
Kappa = (P_o - P_e) / (1 - P_e)     # 0 if P_e = 1
```

`row_sum_i = sum_j CM[i,j]`, `col_sum_i = sum_j CM[j,i]`, `N = sum(CM)`.

**Why:** Paper Table 6; chance-corrected; shows 6c P5 gain vs A4 clearly (κ 0.44 → 0.58).

---

## 13. Optimiser / schedule (A4 and U-TAE defaults)

| Piece | Setting | Why |
|-------|---------|-----|
| Loss | Weighted CE, `ignore_index=255` | §5 |
| Optimizer | **Adam**, `lr=1e-3` | Paper / U-TAE default |
| LR schedule | `ReduceLROnPlateau` (mode max, factor **0.1**, patience **5**) | On val monitor |
| EarlyStopping | patience **20** | No val improvement |
| Monitor | **val W-F1** (default) | Paper-facing |
| U-TAE batch | **2** (~3369 train patches → **~1685 batches/epoch**) | VRAM |
| A4 | Often batch 2; 10c sbatch may use batch 8 (+ optional grad accum toward paper ~16) | PROTOCOL / sbatch |

Plot label **train loss (mean/batch)** = mean of per-batch wCE over all training batches that epoch.

---

## 14. Quick map: name in report → section

| In Phase 2 / JSON / prose | Section |
|---------------------------|---------|
| Softmax / class probs | §1 |
| ReLU / GroupNorm / BN | §2 |
| U-TAE forward, L-TAE Softmax, entropy | §3 |
| CE / Weighted CE | §4-§5 |
| Multinomial logistic regression (P3) | §6 |
| Confusion matrix | §7 |
| Precision, Recall, Sens, Spec, F1 | §8 |
| W-F1, W-P, W-R, … | §9 |
| Accuracy | §10 |
| Mean F1 | §11 |
| Kappa | §12 |
| Adam / plateau / early stop / mean/batch | §13 |

---

## 15. What we did *not* use (frozen baselines)

| Method | Status |
|--------|--------|
| Focal / Dice / Tversky loss | Not used (kept paper-style wCE) |
| Softmax temperature scaling at test | Not used |
| Hierarchical `L_fine + λ L_coarse` | Planned Priority 2 only |
| Attention-weighted CE | Planned Priority 3 only |
| Probe RandomForest in reported tables | Code exists; reports used multinomial LR |

---

*Created 2026-09-08; checked against code 2026-09-08.  
Truth: `multisenge_seg/metrics.py`, `multisenge_utae/models/{utae,ltae}.py`, `probe_layers.py`, `train.py`.*
