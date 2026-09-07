# Publishable Novelty on MultiSenGE - Stay on U-TAE

---

## 0. Baseline status (frozen - updated 2026-09-07)

All numbers = **test tile 31UEQ** unless noted. Protocol = same geographic split / 4-date S1+S2 / shared metrics.

### 6-class


| Model                    | W-F1                    | Kappa      | Notes                                    |
| ------------------------ | ----------------------- | ---------- | ---------------------------------------- |
| Paper ConvLSTM+Inception | 0.9018                  | 0.4186     | RS 2023 Table 5                          |
| A4 ConvLSTM+Inception    | 0.9037                  | 0.4424     | frozen replicate                         |
| U-TAE P4 head            | 0.9012                  | 0.4033     | ~tie A4 on W-F1                          |
| **U-TAE P5 full**        | **0.9387**              | **0.5757** | beats paper + A4                         |
| P3 linear probes (val)   | L2 best W-F1 **0.7477** | -          | multinomial logistic regression; L3 weak |


Class 1 (Dense Built-Up) still below A4 after P5. Classes 2/3/5/6 above A4.

### 10-class


| Model                    | W-F1       | Kappa      | Notes                                            |
| ------------------------ | ---------- | ---------- | ------------------------------------------------ |
| Paper ConvLSTM+Inception | **0.8851** | **0.7945** | RS 2023                                          |
| A4 ConvLSTM+Inception    | 0.8711     | 0.7588     | frozen replicate                                 |
| U-TAE P4 head            | 0.8322     | 0.6971     | clearly below A4/paper                           |
| **U-TAE P5 full**        | **0.8811** | **0.7795** | **beats A4**; near paper (−0.004 W-F1, −0.015 κ) |
| P3 linear probes (10c)   | L1 best W-F1 **0.8009** | 0.6885     | job 100505; L3 weakest 0.5335            |


**Honest bake-off takeaway for novelty:**

- 6c: stock U-TAE + P5 already beats paper on W-F1/kappa → room for a paper is **mechanism + UF minority classes**, not “first time U-TAE wins.”
- 10c: P5 beats **your** A4 but does **not** clearly beat the **published** paper → stronger motivation for a **named** fix (fusion / hierarchy / confusion), not more epochs.

---



## 1. Real research gaps (MultiSenGE + U-TAE world)



### Gap A - Urban fabric is still hard at 10 m

Paper + your runs: Dense vs Sparse Built-Up confuse; class 4 (vegetative specialized) near-zero; vegetation fraction drives UF labels, not sharp edges.

**Your evidence:**

- 6c P5: class 1 still below A4 (≈0.408 vs 0.489).
- 10c P5: class 4 F1 still ~0.16; water precision low; grassland recall weak; Dense F1 ~0.48.



### Gap B - Extreme imbalance + wrong objective

Inverse-frequency CE helps a bit; it does **not** fix confusable pairs. Literature still mostly uses weighted CE. Weighted metrics on 10c are dominated by arable/forest.

### Gap C - Naive S1∥S2 concat

You feed `10 S2 + 2 S1` as one tensor. U-TAE’s L-TAE treats channels as one stream. **Modality-specific temporal behavior** (SAR weather-proof vs optical phenology) is under-modeled on MultiSenGE outside the authors’ ConvLSTM+Inception line.

### Gap D - (new from your runs) Head-only vs full FT asymmetry on 10c

On 6c, P4 ≈ A4; on 10c, P4 collapses (−0.04 W-F1 vs A4). Encoder features alone + linear/decoder head struggle more when the “other” bag is split. Supports **hierarchical UF vs rest** or **confusion-aware** learning as novelty, not only dual-stream fusion.

---



## 2. Publishable directions (U-TAE only) - ranked



### ★ Priority 1 - Modality-aware U-TAE (strongest paper story)

**Gap:** Concat fusion ignores that S1 and S2 need different temporal encoding.

**Your contribution (name it):** e.g. **MA-UTAE / Dual-stream L-TAE**

- Two lightweight encoders (or split channels) → **separate temporal attention** for S1 and S2 → fuse before / inside decoder (gate or FiLM).
- Ablations: S1-only, S2-only, concat U-TAE (current P5), dual-stream.

**Why publishable:** Clear architectural novelty *inside* U-TAE family; matches MultiSenGE’s multimodal claim better than “we ran U-TAE.”

**Success metric:** Beat your **P5 concat** on **urban F1 + kappa** (6c and 10c), especially class 1/4/5; ideally match or beat paper on 10c W-F1/kappa.

---



### ★ Priority 2 - Confusion-aware / hierarchical loss

**Gap:** Weighted CE ≠ separability; Dense↔Sparse, class4↔natural. 10c P4 weakness when “other” is split.

**Your contribution:**

- **Pairwise confusion penalty** from a frozen A4 or early U-TAE CM (reweight gradients on known confuse pairs).
- Or **hierarchical head**: Urban fabric (1-5) vs non-urban, then fine UF head (multi-task on same U-TAE backbone).

**Why publishable:** Loss / task design is yours; backbone stays U-TAE. Fits Gap D from your 10c results.

**Success metric:** Lift class 1 and 4 without killing W-F1; shrink Dense↔Sparse off-diagonal.

---



### ★ Priority 3 - Attention-guided or probe-guided fine-tuning (breast as *inspiration*, rule is yours)

**Gap:** Blind full FT; breast schedule is generic. You already have P3 L0-L3 (6c + 10c).

**Your contribution (must be specific):**

- From P3: freeze L0-L1, unfreeze L2→L3→decoder in stages **decided by probe Δ**.
- Or use **L-TAE attention maps** as soft spatial weights in the loss (focus rare UF pixels).

**Why publishable:** Only if you define a **reproducible selection rule** and ablate vs “full FT from epoch 0” and “breast head→full.”

**Don’t title the paper** “breast TL for EO.”  
**Better title shape:** *Probe-/attention-guided staged U-TAE fine-tuning for imbalanced UF*.

**Note:** P3 alone is **support**, not the paper. Use probe tables as evidence for *where* to intervene.

---



## 3. Detailed solutions - P1 / P2 / P3

**How to use this section:** Read each novelty as a full proposal (why → what → how → ablations → success). Decide **one main** claim (recommended **P1**), optionally **one support** (P2 or P3). Do **not** present all three as equal main contributions.

**Shared baseline (already frozen; do not retrain as “novelty”):**

- Protocol: geographic tiles; train ≈3369 / val ≈1911 / test ≈610; 4 dates Jul-Nov; metrics in `multisenge_seg/metrics.py`.
- Input today: `B×T×12×256×256` with channels **0-9 = S2**, **10-11 = S1 (VV, VH)** via `stack_modalities()` in `multisenge_utae/data.py`. B=batch size, T=time step
- Model today: single-stream `UTAE` (`models/utae.py`) + `LTAE2d` at 1/8 bottleneck (`models/ltae.py`); loss = **weighted CE** (inverse frequency).
- Report baselines every table: **A4 ConvLSTM**, **concat U-TAE P5** (6c W-F1 0.9387 / κ 0.5757; 10c W-F1 0.8811 / κ 0.7795).

---



### 3.1 Priority 1: MA-UTAE (Modality-Aware / Dual-stream U-TAE)



#### Why choose this (reason for novelty)


| Item                     | Detail                                                                                                                                                                                                                                                                                                           |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Research gaps**        | **Primarily Gap C** (naive S1∥S2 concat). **Secondary:** Gap A (better multimodal features can help UF), Gap B (better features help minority classes if fusion is informative).                                                                                                                                 |
| **Problem today**        | One `in_conv` sees 12 mixed channels; one L-TAE attends a **joint** 128-d bottleneck. S1 (weather-robust SAR) and S2 (optical phenology) never get their own temporal encoding. Paper’s ConvLSTM line *does* use separate S1/S2 LSTMs (`multisenge_seg/model.py`) - we currently throw that inductive bias away. |
| **Why publishable**      | Named **architecture** change *inside* the U-TAE family; clear story for MultiSenGE’s multimodal claim; not “we swapped backbone” or “we used breast TL.”                                                                                                                                                        |
| **Why best for a paper** | Reviewers recognize fusion modules; ablations are clean (S1-only / S2-only / concat / dual). Matches strongest default in §2.                                                                                                                                                                                    |




#### Goal / success criteria

- Beat **concat U-TAE P5** on test **kappa** and **urban per-class F1** (classes 1-5), 6-class first then 10-class.
- Ideally reach or beat **paper 10c** W-F1 0.8851 / κ 0.7945 (nice-to-have, not the only claim).
- Show dual-stream > concat and > single-modality under **identical** split/metrics.



#### Method: what we build

**Name:** **MA-UTAE** (Modality-Aware U-TAE) / Dual-stream L-TAE.

**High-level idea:** Keep U-TAE’s spatial U-Net + L-TAE idea, but run **two** modality streams until (or through) temporal attention, then **fuse** into one decoder.

**Recommended design (implementable in our repo):**

```
s2: B×T×10×H×W ──► Encoder_S2 (narrow U-TAE downs) ──► feat_S2 @ 32×32
s1: B×T×2×H×W  ──► Encoder_S1 (narrow)            ──► feat_S1 @ 32×32
                         │                                  │
                    L-TAE_S2 (own attn)                L-TAE_S1 (own attn)
                         │                                  │
                         └──── Fusion (concat + 1×1 / gate / FiLM) ────►
                                      │
                                      ▼
                         Shared U-TAE decoder + skips
                         (skips: either fused early or dual TemporalAggregator)
                                      │
                                      ▼
                              out_conv → logits
```

**Concrete options (pick one with sir; A is default):**


| Variant                                                | Description                                                                                | Pros / cons                                    |
| ------------------------------------------------------ | ------------------------------------------------------------------------------------------ | ---------------------------------------------- |
| **A. Dual encoder + dual L-TAE + late fuse (default)** | Separate `in_conv`+downs per modality; two `LTAE2d`; fuse bottleneck maps; **one** decoder | Strongest story; more params                   |
| **B. Shared spatial encoder, split channels at L-TAE** | One encoder on 12ch; split/projected S2 vs S1 features into two L-TAEs; fuse               | Lighter; weaker “modality-aware encoder” claim |
| **C. Dual encoder, single shared L-TAE**               | Fuse before one L-TAE                                                                      | Weaker; less novelty                           |


**Fusion block (choose + ablate):**

1. **Concat + 1×1 conv** (simplest baseline fusion).
2. **Gated fusion:** `σ(W[f_s2;f_s1]) ⊙ f_s2 + (1−σ) ⊙ f_s1` (or FiLM: S1 modulates S2).
3. Optional: fuse also at skip levels (heavier).

**Reuse from our code:**

- Dataset already returns separate `"s2"` / `"s1"` in `collate_utae`  -  only training loop must stop relying solely on fused `"x"` (or build `x_s2`/`x_s1` inside the model from `x[:, :, 0:10]` / `x[:, :, 10:12]`).
- Copy/adapt `UTAE`, `LTAE2d`, `set_train_mode`, train/eval sbatch pattern.
- A4’s separate S1/S2 LSTMs are the **motivation**, not the implementation (we stay U-TAE).



#### Training protocol (same as now)

1. Norm stats from train (existing).
2. Optional: **P4 head-only** on new decoder/fusion (freeze early dual encoders) → **P5 full FT** from that ckpt (same schedule that worked).
3. Loss: start with **same weighted CE** so gains are attributed to architecture, not a new loss (loss changes = P2).
4. Start **6-class** (faster; already beat paper) → then **10-class**.
5. PARAM: new `train_ma_utae_*.sbatch`, same exclude nodes / GPU partition.



#### Mandatory ablations (paper table)


| Run                   | What                    |
| --------------------- | ----------------------- |
| Concat U-TAE P5       | Current frozen baseline |
| S2-only U-TAE         | Drop S1                 |
| S1-only U-TAE         | Drop S2 (expect weak)   |
| MA-UTAE (concat fuse) | Dual stream + 1×1       |
| MA-UTAE (gated fuse)  | Dual stream + gate      |
| A4 ConvLSTM           | Already have            |




#### Risks / mitigation

- More VRAM → smaller batch or narrower widths (`encoder_widths` reduce for each stream).
- Overfitting dual path → stronger weight decay / dropout in fusion; keep P4→P5.
- If W-F1 flat but urban F1↑ → still publishable if we **pre-register** urban F1 + kappa as primary (Gap A).



#### Effort (rough)

- Code skeleton: ~1-2 weeks; 6c full train+ablations: ~1-2 weeks PARAM; 10c: another ~1-2 weeks. Sir-facing: architecture figure + ablation table.



#### What you tell sir in one sentence

> “We replace early S1∥S2 concat with modality-specific U-TAE streams and separate L-TAE temporal attention, then fuse, targeting Gap C, evaluated against our frozen concat U-TAE P5 under the exact MultiSenGE geographic protocol.”

---



### 3.2 Priority 2: Confusion-aware loss / hierarchical UF head



#### Why choose this (reason for novelty)


| Item                    | Detail                                                                                                                                                                                               |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Research gaps**       | **Primarily Gaps A, B, D.** Dense↔Sparse / class4 (A); weighted CE ≠ pair confusion (B); 10c P4 collapse when “other” is split (D).                                                                  |
| **Problem today**       | Weighted CE upweights rare classes globally but does **not** say “stop confusing 1 with 2.” CM shows structured mistakes. 10c P4 W-F1 0.832 ≪ A4 shows decoder+head struggle when taxonomy is finer. |
| **Why publishable**     | **Loss / task design** is yours; backbone stays stock U-TAE (or MA-UTAE later). Method papers accept confusion / hierarchical multi-task designs.                                                    |
| **When better than P1** | If sir prefers **UF class quality** over multimodal architecture; or as **add-on** after P1.                                                                                                         |


**Two sub-methods (can be separate papers or one paper with two modules; prefer pick ONE as main):**

---



#### 3.2.A Hierarchical UF multi-task head (recommended P2 variant)

**Idea:** Same U-TAE backbone; two heads:

1. **Coarse head:** binary (or 2-way) **Urban fabric (classes 1-5) vs Rest**.
2. **Fine head:** full K-way logits (6 or 10 classes) as now.

**Loss:**

`L = L_fine(wCE) + λ * L_coarse(CE)`

- `L_fine(wCE)` = weighted cross-entropy on the full 6/10-class head  
- `L_coarse(CE)` = cross-entropy on the urban-vs-rest head  
- `λ` = weight for the coarse task (tuned on validation)

Optional: only apply fine UF loss on pixels where coarse = urban (or use soft weighting).

**Why it fits Gap D:** Coarse task is easy and stable (like 6c “class 6 bag”); fine head specializes on UF; mirrors why 6c was easier than 10c for P4.

**Implementation hooks:**

- Backbone: existing `UTAE` up to last decoder feature (before / parallel to `out_conv`).
- Add `out_conv_coarse` (2 classes) + keep `out_conv` (K classes).
- Labels: map mask → coarse `{0: rest, 1: UF}` for classes 1-5.
- Train: full FT from concat P5 init (faster) or head→full.
- Inference: use fine head only (or gate fine with coarse).

---



#### 3.2.B Pairwise confusion-aware penalty (CM-guided)

**Idea:** From a **frozen** confusion matrix (A4 or our U-TAE P5 on **val**, not test; avoid leakage):

- Identify top confuse pairs, e.g. (1,2), (4,6), (2,3).
- Add a penalty when the model assigns high probability to the wrong member of a known pair.

**Example loss term (one clean option):**

- Soft confusion loss: for pixels with true class c, extra weight on logit competition with confused class c' (pairwise logistic / margin), **or**
- Reweight CE with a **class-pair matrix** W_{c,c'} derived from normalized CM off-diagonals (fixed schedule, not trained on test).

**Implementation hooks:**

- Export CM from existing `test_metrics.json` / recompute on **val** with `evaluate()`.
- Config YAML: list of pairs + λ.
- Keep architecture = stock U-TAE P5; change `criterion` in `train.py` only.

**Sir caveat:** Must freeze CM from val (or train) and state that clearly; do not tune pairs on test 31UEQ.

---



#### Goal / success criteria (P2)

- Lift **class 1 and class 4 F1** vs concat P5 without dropping W-F1 more than a small tolerance (e.g. ≤0.005).
- Shrink Dense↔Sparse off-diagonal in CM.
- 10c: improve κ and UF classes; ideally close paper gap.



#### Mandatory ablations


| Run                                        | What                              |
| ------------------------------------------ | --------------------------------- |
| U-TAE P5 + weighted CE only                | Baseline                          |
| + hierarchical coarse (vary λ)             | 3.2.A                             |
| + confusion pairs (vary λ / pair set)      | 3.2.B                             |
| Hierarchical **without** weighted CE boost | Show hierarchy ≠ just reweighting |




#### Risks / mitigation

- λ too high → majority collapse or weird precision/recall tradeoffs → tune on **val**.
- Hierarchical may inflate “urban vs rest” without fixing Dense vs Sparse → still need pair term or stronger UF sampling.
- Weaker as **sole** paper vs P1 unless UF gains are large and well ablated.



#### Effort (rough)

- Hierarchical head: ~3-5 days code + 1 week 6c runs.
- Confusion loss: ~2-4 days + runs.
- Faster than P1; good MTech “method chapter” even if paper prefers P1.



#### What you tell sir in one sentence

> “We keep U-TAE fixed and add a hierarchical urban-vs-rest head and/or a validation-CM-guided pairwise confusion penalty to attack Gaps A/B/D: Dense/Sparse/class4 errors that weighted CE does not fix.”

---



### 3.3 Priority 3: Probe-guided / attention-guided fine-tuning



#### Why choose this (reason for novelty)


| Item                           | Detail                                                                                                                                                                                              |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Research gaps**              | Supports **A, B, D** (where to train / which pixels); weak alone for **C**. Uses our **P3 L0-L3** evidence (6c: L2 best, L3 weak).                                                                  |
| **Problem today**              | P4 freezes encoder blindly; P5 unfreezes everything. Breast schedule is generic. Probes show **not all layers are equal**. L-TAE attention exists (`attn` from `LTAE2d`) but is unused in the loss. |
| **Why publishable only if…**   | You define a **reproducible rule** (not “we tuned by hand”) and beat naive full FT + beat head→full.                                                                                                |
| **Why usually not main claim** | Easy for reviewers to call it “training curriculum.” Better as **support** under P1 or P2.                                                                                                          |




#### Two concrete rules (pick one to implement cleanly)

---



#### 3.3.A Probe-Δ staged unfreezing (recommended P3 variant)

**Rule (example: write exactly in paper):**

1. Run linear probes L0-L3 on **val** (already have recipe in `probe_layers.py`).
2. Rank levels by val W-F1 (6c: L2 > L1 ≈ L0 ≫ L3).
3. Stage training:
  - **Stage 0:** train decoder/head only (like P4).
  - **Stage 1:** unfreeze levels with probe W-F1 ≥ τ (e.g. L2 block) + decoder.
  - **Stage 2:** unfreeze next band (L1) …
  - **Stage 3:** full model (including L-TAE / L3) at lower LR.
4. τ and stage lengths fixed a priori or selected on **val** only.

**Implementation:**

- Extend `UTAE.set_train_mode` with modes: `head`, `unfreeze_L2`, `unfreeze_L1_L2`, `full`.
- Map modules: L0=`in_conv`, L1=`down_blocks[0]`, L2=`down_blocks[1]`, L3=`down_blocks[2]`+`temporal_encoder`.
- Init from ImageNet-random or from P4 ckpt; log probe table as justification figure.

---



#### 3.3.B Attention-weighted loss (pixel focus)

**Idea:** Use L-TAE attention over time (and/or spatial attention energy) to upweight loss on pixels that are “hard” or rare UF.

- Example:  w(h,w) \propto 1 + \alpha \cdot \mathrm{entropy}(\mathrm{attn}_{:,h,w})  or upweight pixels whose GT class ∈ {1,4,5}.
- Or: weight by inverse attention on majority class regions (force model to care about low-attention UF).

**Implementation:**

- `temporal_encoder` already returns `att` (`n_head×B×T×H×W`); upsample to 256×256; detach weights (do not backprop through attn for stability, or do; ablate).
- Modify `train_one_epoch` CE to `reduction='none'` then multiply by `w`.



#### Goal / success criteria (P3)

- Beat **P5 from-scratch full** and beat **standard head→full** on urban F1 / kappa with **same** epochs/LR budget.
- Show stages follow probe ranking (ablate: reverse order should be worse).



#### Mandatory ablations


| Run                              | What           |
| -------------------------------- | -------------- |
| Full FT from epoch 0             |                |
| Head→full (current breast-style) | Current P4→P5  |
| Probe-staged (forward order)     | 3.3.A          |
| Probe-staged (reverse order)     | Sanity         |
| + attention loss weights         | 3.3.B optional |




#### Risks / mitigation

- Main-claim risk: “just curriculum learning.” Mitigate with **fixed rule** + reverse-order ablation.
- Attention weighting can destabilize → start with detached attn, small α.
- Alone may not beat paper on 10c; combine with P1/P2 for headline numbers.



#### Effort (rough)

- Staging API: ~2-4 days; full compare runs: ~1 week 6c.
- Cheapest novelty engineering; weakest standalone paper story.



#### What you tell sir in one sentence

> “We replace blind head→full fine-tuning with a probe-Δ rule (and optional L-TAE attention loss weights) so unfreezing order is justified by our L0-L3 diagnostics; support for Gaps A/B/D, not a fusion claim.”

---



### 3.4 How to combine (for the decision meeting)


| Choice       | Verdict                                    | Gaps covered  | Notes for sir                                                                |
| ------------ | ------------------------------------------ | ------------- | ---------------------------------------------------------------------------- |
| **P1 alone** | **Best single paper**                      | C (+A)        | Strongest novelty; do this if only one.                                      |
| **P2 alone** | Good method paper / strong MTech           | A, B, D       | Faster; UF-focused; weaker multimodal story.                                 |
| **P3 alone** | **Not recommended as sole novelty**        | A/B/D support | Curriculum; keep as ablation.                                                |
| **P1 + P2**  | **Best duo**                               | C + A/B/D     | Main = MA-UTAE; hierarchical/confusion as second module/ablation.            |
| **P1 + P3**  | Good duo                                   | C + training  | Main = MA-UTAE; staged FT as training ablation.                              |
| **P2 + P3**  | OK duo, no fusion                          | A/B/D         | If rejecting architecture change.                                            |
| **P1+P2+P3** | Possible engineering, **bad single claim** | All           | Too many moving parts; hard to ablate; don’t pitch as three equal novelties. |


**Recommended pitch to sir:**

1. **Main:** P1 MA-UTAE (Gap C).
2. **Optional add-on:** P2 hierarchical head (Gaps A/D) **or** P3 probe staging as ablation.
3. Keep concat U-TAE P5 + A4 as frozen baselines in every table.

---



### 3.5 Decision checklist (fill after meeting)

- [ ] Main novelty: **P1** / **P2** / **P3**
- [ ] Support novelty (optional): **P2** / **P3** / none
- [ ] P2 flavor if chosen: hierarchical head / confusion pairs / both
- [ ] Start taxonomy: **6-class first** / 10-class first
- [ ] Must beat paper 10c W-F1? **Yes** / **No** (beat A4 + UF classes enough)
- [ ] Timeline / PARAM budget: ________

---

*Section 3 added 2026-09-07 for sir discussion before implementation.*