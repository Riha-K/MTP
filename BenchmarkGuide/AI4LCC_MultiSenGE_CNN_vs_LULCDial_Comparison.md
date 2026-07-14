# AI4LCC MultiSenGE / MultiSenNA — Prior CNN Results vs LULCDial-S1

> **Purpose:** Answer the supervisor question: *“AI4LCC is used in CNNs — are your results better? Is that novelty enough?”*  
> **Date:** 2026-07-13  
> **Your numbers:** GE example F1 ZS **0.019** → FT **0.799**; NA transfer (no NA train) **0.670**  
> **PDF in this folder:** [`multiSenge_AI4LCC.pdf`](multiSenge_AI4LCC.pdf) (ISPRS Annals 2022 MultiSenGE paper)

---

## 1. One-sentence answer for your professor

**You should not claim “our F1 0.80 beats their U-Net 0.74.”**  
Those papers do **pixel semantic segmentation** (often **S2 or S1+S2**, urban-focused or merged classes).  
You do **patch-level multi-label classify + dialogue** with a **VLM**, **Sentinel-1 VH only**, **all 14 OCSGE names**.  

**What is fair to claim:**  
same **AI4LCC labels/geography**, **different task and output**; first **instruction/VLM protocol** on this data; CNNs already mapped pixels — you make the model **talk** about OCSGE classes and **transfer GE→NA without NA fine-tune**.

---

## 2. Critical: apples vs oranges (read this first)

| Axis | Prior AI4LCC CNN papers | **Your LULCDial-S1** |
| ---- | ----------------------- | -------------------- |
| **Task** | Semantic **segmentation** (label **every pixel**) | **Patch-level** multi-label set + **2-turn dialogue** (text) |
| **Output** | Class map / raster | Natural-language class lists |
| **Sensors** | Mostly **S2 IRRG**, or **S1+S2** multi-date | **S1 VH only**, **one** median date |
| **Classes** | Often **urban 1–5 + other**, or **6 / 10** merged | Official **14** OCSGE names (NA: +15 Beaches) |
| **Metric** | Weighted F1 / Accuracy / Kappa on **pixels** | **example F1** + exact **set-match** on **patches** |
| **Model family** | U-Net, ConvLSTM+Inception (**CNN**) | EarthDial_4B_MS → LULCDial-S1 (**VLM**) |
| **Split** | Geographic tile splits (e.g. Metz / Strasbourg tiles) | Your 90/10 MD5 patch split → **801** GE val patches |

So: **same underlying French OCSGE / AI4LCC family**, **not** the same leaderboard row.

---

## 3. What the MultiSenGE PDF / ISPRS 2022 paper actually reports

**Paper:** Wenger et al., *MultiSenGE: A Multimodal and Multitemporal Benchmark Dataset…*, ISPRS Annals V-3-2022, 635–640.  
https://doi.org/10.5194/isprs-annals-v-3-2022-635-2022  
Local PDF: `BenchmarkGuide/AI4LCC/multiSenge_AI4LCC.pdf`

### Setup (baseline in the dataset paper)

| Item | Value |
| ---- | ----- |
| Focus | **Urban** semantic segmentation (classes **1–5** + aggregated other) |
| Area | **Metz**, Sentinel-2 tile **T31UGQ** |
| Input | Single-date **S2 IRRG** (± indices for second model) — **not** full S1 LULC VLM |
| Models | U-Net + VGG-16 |

### Numbers (Metz test)

| Model | Weighted F1 |
| ----- | ----------- |
| **U-Net-IRRG** (ImageNet init) | **0.7364** |
| U-Net-Index (IRRG + NDVI/NDBI/eNDVI, random init) | 0.7214 |
| Example hard classes | Dense built-up F1 ≈ **0.52**; roads (class 5) ≈ **0.59** (IRRG) |

**Takeaway from PDF:** even with **optical** S2, fine urban classes are hard. The paper’s main gift is the **dataset**; heavy multitemporal S1+S2 CNNs came in follow-ups.

---

## 4. Other published uses of MultiSenGE (CNN — not VLM)

### 4.1 Remote Sensing 2023 — ConvLSTM + Inception on MultiSenGE

**Paper:** Wenger et al., *Multimodal and Multitemporal LULC Semantic Segmentation on S1 and S2…*, Remote Sensing 15(1):151, 2023.  
https://doi.org/10.3390/rs15010151

| Item | Detail |
| ---- | ------ |
| Model | **ConvLSTM+Inception-S1S2** (U-Net-style + temporal + spectral fusion) |
| Input | **Multi-date Sentinel-1 + Sentinel-2** |
| Classes | Experiments with **6** and **10** (some OCSGE classes **merged**) |
| Reported highlight | Best **6-class** weighted F1 up to ≈ **0.90** (ConvLSTM-S1S2 / +Inception variants in their tables) |
| 10-class | Lower global weighted F1 (harder taxonomy; paper reports e.g. ~**0.64** weighted F1 range for best 10-class setups — see their Table 7) |

Still **pixel segmentation**, multimodal/temporal — **not** comparable one-to-one with your patch example F1.

### 4.2 JURSE / thesis follow-ups

Wenger PhD and related Unistra work continue CNN inference / LULC mapping on MultiSenGE (segmentation tooling). Same family: **CNN maps**, not dialogue VLMs.

---

## 5. MultiSenNA — where it was used

### 5.1 Dataset role

| Collection | Region | Patches | Role in AI4LCC |
| ---------- | ------ | ------: | -------------- |
| MultiSenGE | Grand-Est | 8,157 | Original benchmark (2022) |
| **MultiSenNA** | Nouvelle-Aquitaine | ~12,258 | Sister region; class **15 = Beaches, Sand** |

Portal: https://doi.theia.data-terra.org/ai4lcc/

### 5.2 Remote Sensing Letters 2025 — CNN cross-region fine-tuning

**Paper:** Wenger, Puissant, Forestier, *Multitemporal and Multimodal Cross-Region Fine-Tuning: MultiSenGE and MultiSenNA Applications*, Remote Sensing Letters, 2025.  
PDF: https://germain-forestier.info/publis/rsl2025.pdf

They use **ConvLSTM+Inception-S1S2** and **fine-tune across regions** (train/pretrain on one, adapt on the other).

**Selected results (their Table 2 — CNN pixel metrics):**

| Setting (simplified) | Region tested | Accuracy | Weighted F1 |
| -------------------- | ------------- | --------: | ----------: |
| Fine-tune all layers, weights from MultiSenGE → **MultiSenNA** (test 5) | NA | **81.90%** | **83.17%** |
| Fine-tune, weights from MultiSenNA → **MultiSenGE** (test 6) | GE | **85.55%** | **86.47%** |
| Urban only (classes 1–5) weighted F1, full data | GE / NA | — | **60.44%** / **53.54%** |

**Contrast with you:**

| Aspect | Wenger RSL 2025 | Your E4 (LULCDial-S1) |
| ------ | --------------- | --------------------- |
| On MultiSenNA | Fine-tune / train on NA (with GE pretrain) | No NA training — GE model only |
| Task | CNN segmentation (pixel maps) | VLM patch dialogue / multi-label text |
| Claim | Transfer learning with NA adaptation reduces carbon vs train-from-scratch | Regional zero-shot transfer of a dialogue VLM |

So MultiSenNA **is** used in literature — as a **CNN fine-tuning** target — **not** as a VLM dialogue bench.

---

## 6. Any VLM papers on MultiSenGE / MultiSenNA?

**Public search (Jul 2026):** no established paper found that:

- builds an **instruction / dialogue** benchmark on AI4LCC OCSGE, or  
- reports EarthDial / LLaVA / GeoChat-style results on **MultiSenGE/NA** as the main LULC chat task.

**EarthDial (CVPR 2025)** uses many RS tasks and **SAR** (ships, change, etc.) and **BigEarthNet-style** optical LULC — **not** this French OCSGE MultiSenGE dialogue protocol.

→ Your **N1** (first VLM instruction + eval protocol on AI4LCC S1 VH + 14-class OCSGE) remains a **defensible gap claim**, with the usual caveat: “to the best of our knowledge.”

---

## 7. Side-by-side: their scores vs yours (for slides — with disclaimer)

| Work | Task | Input | Metric | Number | Comparable to you? |
| ---- | ---- | ----- | ------ | -----: | ------------------ |
| ISPRS 2022 U-Net-IRRG | Urban seg (Metz) | S2 IRRG | Weighted F1 | **0.74** | **No** |
| RS 2023 ConvLSTM+Inception | LULC seg (6-cls highlight) | S1+S2 multi-date | Weighted F1 | ~**0.90** | **No** |
| RSL 2025 FT on NA | LULC seg | S1+S2 multi-date | Acc / wF1 | **81.9% / 83.2%** | **No** (and they **train on NA**) |
| **LULCDial-S1 GE** | Patch multi-label + dialogue | **S1 VH** | example F1 | **0.799** | Your primary |
| **LULCDial-S1 NA** | Same, **no NA FT** | **S1 VH** | example F1 | **0.670** | Your transfer |

**If professor insists on “compare performance”:**  
compare **within your protocol** (ZS vs FT vs NA), and say CNNs are **complementary baselines for mapping**, not the same score. Optional future work: report a **CNN patch-level multi-label** baseline on **your exact 801 JSONL** for a fairer number — that would be a real apples-to-apples addition.

---

## 8. What you did that is **different** (novelty framing)

| # | What you did | Why it is not “just another MultiSenGE CNN” |
| - | ------------ | --------------------------------------------- |
| **1** | Converted AI4LCC into **EarthDial-style QA** (classify + 2-turn dialogue) | New **protocol / benchmark**, not a U-Net |
| **2** | Showed EarthDial ZS ≈ **floor** (0.02) then FT ≈ **0.80** on GE | Proves **VLM adaptation** is needed for this SAR LULC chat task |
| **3** | Evaluated **GE→NA without training on NA** | Different from RSL 2025, which **fine-tunes on NA** |
| **4** | Kept **14 official OCSGE names** (no remap to 7 bare classes) | Aligns with expert labels; dialogue-ready |
| **5** | Sensor constraint: **S1 VH only** | Harder / different than S2 or S1+S2 fusion CNNs |

**Worth mentioning as novelty:** yes — as **task + protocol + VLM evidence + zero-shot regional transfer**, **not** as “we beat U-Net 0.74 with 0.80.”

---

## 9. Suggested 45-second reply in the meeting

> “MultiSenGE/NA are already used for **CNN semantic segmentation** — U-Net about **0.74** weighted F1 on urban Metz with S2, and later ConvLSTM S1+S2 papers report much higher pixel F1, and RSL 2025 fine-tunes CNNs **on** MultiSenNA.  
> I am **not** competing on that pixel leaderboard. I built the first **VLM instruction benchmark** on the same OCSGE labels with **S1-only** dialogue, raised EarthDial from **0.02 → 0.80** example F1 on GE val, and got **0.67** on MultiSenNA **without** NA training.  
> So the novelty is **interactive SAR LULC understanding and transfer**, complementary to existing CNN mappers.”

---

## 10. References (quick)

1. Wenger et al. (2022) MultiSenGE dataset paper — ISPRS Annals.  
2. Wenger et al. (2023) ConvLSTM+Inception on MultiSenGE — Remote Sensing 15(1):151.  
3. Wenger et al. (2025) MultiSenGE ↔ MultiSenNA CNN fine-tuning — Remote Sensing Letters.  
4. AI4LCC portal — https://doi.theia.data-terra.org/ai4lcc/  
5. EarthDial (CVPR 2025) — RS VLM; **not** AI4LCC OCSGE dialogue.  
6. Your metrics — `LULCDial-s1/data/baresoil_s1/metrics/v0.1/`

---

*Internal note: if you later need a fair numeric “vs CNN” column, train a simple multi-label CNN/MLP on the **same** `ai4lcc_val.jsonl` patch targets and report example F1 — do not reuse pixel weighted F1 from Metz.*
