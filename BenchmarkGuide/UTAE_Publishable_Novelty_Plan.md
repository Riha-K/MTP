# Publishable Novelty on MultiSenGE — Stay on U-TAE

**Audience:** You + professor (decide later).  
**Context:** Breast TL + stock U-TAE is a solid **MTech chapter**; it is **not** a paper contribution by itself. Reviewers will say: baseline swap + known training schedule.  
**Constraint:** Stick to **U-TAE** (no SegFormer). Novelty = a **named module / loss / protocol / fusion rule** you designed.

**Related:** `ROADMAP.md` · `multisenge_utae/RESULTS_UTAE_6CLASS_FULL.md` · `BenchmarkGuide/MultiSenGE_Validation_and_VLM_Plan.md`

**Status snapshot (6-class, test 31UEQ):**
| Model | W-F1 | Kappa |
|-------|------|-------|
| A4 ConvLSTM+Inception | 0.9037 | 0.4424 |
| U-TAE P4 head | 0.9012 | 0.4033 |
| **U-TAE P5 full** | **0.9387** | **0.5757** |

Class 1 (Dense Built-Up) still below A4. 10-class U-TAE head/full in progress.

---

## 1. What is *not* publishable as the main claim

| Claim | Why it fails peer review |
|-------|---------------------------|
| “We used breast TL on EO” | Method already published in medical imaging |
| “We used U-TAE on MultiSenGE” | U-TAE is known; MultiSenGE papers already exist from the dataset authors |
| “W-F1 went up” alone | Incremental bake-off unless tied to a **new mechanism** |

Keep those as **supporting experiments**. Put novelty in a **named module / protocol / loss / fusion rule** that you designed.

**Honest framing for sir:**
> We do not claim a new TL algorithm or invent U-TAE. We show that a modern temporal model under a paper-matched protocol improves MultiSenGE LULC, and we propose **[ONE named mechanism]** to address a documented gap (fusion / UF confusion / geo shift).

---

## 2. Real research gaps (MultiSenGE + U-TAE world)

### Gap A — Urban fabric is still hard at 10 m

Paper + your runs: Dense vs Sparse Built-Up confuse; class 4 (vegetative specialized) near-zero; vegetation fraction drives UF labels, not sharp edges.

**Your evidence:** 6c P5 beat A4 overall, but class 1 still below your ConvLSTM.

### Gap B — Extreme imbalance + wrong objective

Inverse-frequency CE helps a bit; it does **not** fix confusable pairs. Literature still mostly uses weighted CE.

### Gap C — Naive S1∥S2 concat

You feed `10 S2 + 2 S1` as one tensor. U-TAE’s L-TAE treats channels as one stream. **Modality-specific temporal behavior** (SAR weather-proof vs optical phenology) is under-modeled on MultiSenGE outside the authors’ ConvLSTM+Inception line.

### Gap D — Geographic shift

JURSE / MultiSenNA papers: train Grand-Est → other cities drops a lot. Your val→test gap already shows tile shift. Few works fix **tile-robust** U-TAE under the same MultiSenGE protocol.

### Gap E — Temporal collapse + fixed 4 dates

U-TAE collapses time to one map (by design). Paper locks Jul/Aug/Sep/Nov. Gaps: which dates matter, how attention uses them, cloudy / missing dates, longer series — mostly studied on **crops**, not MultiSenGE UF.

### Gap F — Author-group monopoly

Almost all MultiSenGE DL results are from the same team (Wenger / Puissant / Forestier et al.). An **independent** strong U-TAE study helps, but only if you add a **method**, not only replication.

### Gap G — CNN vs language/SAR dialogue unused

You already have LULCDial. Almost nobody closes the loop: **pixel U-TAE errors ↔ patch-level dialogue labels**.

---

## 3. Publishable directions (U-TAE only) — ranked

### ★ Priority 1 — Modality-aware U-TAE (strongest paper story)

**Gap:** Concat fusion ignores that S1 and S2 need different temporal encoding.

**Your contribution (name it):** e.g. **MA-UTAE / Dual-stream L-TAE**
- Two lightweight encoders (or split channels) → **separate temporal attention** for S1 and S2 → fuse before / inside decoder (gate or FiLM).
- Ablations: S1-only, S2-only, concat U-TAE (current), dual-stream.

**Why publishable:** Clear architectural novelty *inside* U-TAE family; matches MultiSenGE’s multimodal claim better than “we ran U-TAE.”

**Success metric:** Beat your P5 concat on **urban F1 + kappa**, especially class 1/4/5.

---

### ★ Priority 2 — Confusion-aware / hierarchical loss (method, not hocus)

**Gap:** Weighted CE ≠ separability; Dense↔Sparse, class4↔natural.

**Your contribution:**
- **Pairwise confusion penalty** from a frozen A4 or early U-TAE CM (reweight gradients on known confuse pairs).
- Or **hierarchical head**: Urban fabric (1–5) vs non-urban, then fine UF head (multi-task on same U-TAE backbone).

**Why publishable:** Loss / task design is yours; backbone stays U-TAE.

**Success metric:** Lift class 1 and 4 without killing W-F1.

---

### ★ Priority 3 — Attention-guided or probe-guided fine-tuning (breast as *inspiration*, rule is yours)

**Gap:** Blind full FT; breast schedule is generic.

**Your contribution (must be specific):**
- From P3: freeze L0–L1, unfreeze L2→L3→decoder in stages **decided by probe Δ**.
- Or use **L-TAE attention maps** as soft spatial weights in the loss (focus rare UF pixels).

**Why publishable:** Only if you define a **reproducible selection rule** and ablate vs “full FT from epoch 0” and “breast head→full.”

**Don’t title the paper** “breast TL for EO.”  
**Better title shape:** *Probe-/attention-guided staged U-TAE fine-tuning for imbalanced UF*.

---

### ★ Priority 4 — Geographic robustness under fixed protocol

**Gap:** Tile holdout + MultiSenNA / city transfer weak.

**Your contribution:**
- Tile-adversarial or CORAL on U-TAE bottleneck; or **test-time style norm** per tile.
- Eval: your 31UEQ + MultiSenNA / extra city if labels exist.

**Why publishable:** Addresses documented MultiSenGE limitation; U-TAE backbone fixed.

---

### Priority 5 — Temporal design on MultiSenGE (careful)

**Gap:** Fixed 4 dates; unknown which dates L-TAE uses.

**Your contribution:**
- Attention rollout / date ablation (drop Aug, etc.).
- Optional: learnable date importance or pad-mask for missing S2.

**Risk:** Easy to look like a minor ablation paper unless tied to UF phenology narrative.

---

### Priority 6 — CNN–VLM consistency (extension track)

**Gap:** Segmentation and VLM live in silos.

**Your contribution:** Consistency loss or error atlas: where U-TAE confuses UF, does LULCDial agree?

**Venue:** workshops / IJRS application; stronger as **second contribution** than sole method paper.

---

## 4. What I would *not* do for a paper

- More epochs / LR only
- “Ensemble A4+U-TAE” as **the** novelty (OK as extra table row, weak as main claim)
- Re-running breast probes as the headline
- Switching to SegFormer (ruled out; also still “another backbone”)

---

## 5. Concrete paper narrative (draft)

> MultiSenGE urban fabric segmentation remains limited by (i) naive multimodal fusion, (ii) confusion among UF classes under severe imbalance, and (iii) geographic shift. We keep U-TAE as the temporal backbone and propose **[MA-UTAE + hierarchical/confusion-aware learning]**, evaluated under the **exact** Wenger geographic protocol against ConvLSTM+Inception and plain U-TAE. We show gains especially on minority UF classes and analyze L-TAE temporal attention / layer probes to explain *why*.

That is a contribution. “We copied breast TL and U-TAE got higher W-F1” is not.

---

## 6. Recommended roadmap (practical)

| Phase | Action |
|-------|--------|
| **Now** | Finish 10c head → full → test (baseline numbers for the paper table) |
| **Next (novelty)** | Implement **dual-stream S1/S2 U-TAE** *or* **UF hierarchical multi-task** — pick **one** as main |
| **Support** | Probe-guided staging + CM-based pair weights as ablations |
| **Optional** | MultiSenNA / city transfer + VLM consistency as discussion |

### Decision checklist (pick later)

- [ ] **Main novelty A:** Dual-stream / modality-aware U-TAE (MA-UTAE)
- [ ] **Main novelty B:** Hierarchical UF multi-task head
- [ ] **Main novelty C:** Confusion-pair loss from CM
- [ ] **Support only:** Probe-guided staging (not paper title)
- [ ] **Support only:** Geo-robustness / MultiSenNA
- [ ] **Extension only:** CNN–VLM consistency

**Best single bet for a paper:** modality-aware U-TAE (separate S1/S2 temporal paths + learned fusion).  
**Second bet:** hierarchical UF multi-task head.

---

## 7. MTech vs paper (honest bottom line)

| Track | What is enough |
|-------|----------------|
| **MTech** | Independent ConvLSTM replicate + U-TAE beat + TL schedule + full metrics/plots — if written carefully |
| **Paper** | Need **one new mechanism** inside/around U-TAE aimed at a **documented MultiSenGE gap** (fusion, UF confusion, or geo shift) |

Breast method = **borrowed schedule**. Your novelty = **fair bake-off + TL diagnostics + (next) one idea that is clearly yours**.

---

## 8. Next engineering step (when you decide)

1. Lock main novelty: **dual-stream fusion** *or* **hierarchical UF head**.
2. Keep current concat U-TAE P5 as **baseline** in every table.
3. Same geographic split, same metrics (`multisenge_seg/metrics.py`).
4. Ablations mandatory: S1 / S2 / concat / proposed; 6-class then 10-class.
5. Do **not** start novelty coding until 10-class head→full→test baseline is frozen (optional but cleaner paper timeline).

---

*Written 2026-09-04 for later decision. Update this file when you lock a novelty path.*
