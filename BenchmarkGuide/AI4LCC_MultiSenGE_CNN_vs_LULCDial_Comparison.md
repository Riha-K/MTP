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

## 6. EarthDial (CVPR 2025) S1 metrics — vs your DataSET tables

**Sources:**  
- Main paper: *EarthDial: Turning Multi-sensory Earth Observations to Interactive Dialogues* (CVPR 2025)  
- Supplemental: same title, supplemental PDF  

**Bottom line:** EarthDial **does** publish S1/SAR numbers, but on **ships (+ QuakeSet prompts)** — **not** AI4LCC MultiSenGE/NA OCSGE LULC. Its **LULC classification** metrics are mainly **RGB** (BigEarthNet **68.82%**) and **S2/MS** (BigEarthNet **69.94%**, LCZ **60.72%**, TreeSatAI **56.61%**). Do **not** put those Acc % or Table 5 ship mAP in the same scoreboard row as your example F1 **0.799**.

### 6.1 What S1 tasks EarthDial reports

| EarthDial S1 / SAR use | Task type | In MultiSenGE/NA? |
| ---------------------- | --------- | ----------------- |
| SAR-Ship [47] | Referred detection + region captioning | No |
| SRSDD-v1.0 [25] | Region captioning (SAR ships) | No |
| QuakeSet [8] | Bi-temporal S1: earthquake? + magnitude (prompts; no clear accuracy table like Table 5) | No |
| Satlas / pretrain S1 | Pretraining QA | No |
| BigEarthNet | Land-cover **classification** | Optical / MS — **not** S1 OCSGE |

Supplemental dataset list confirms S1 downstream sets = **SAR-Ship, SRSDD, QuakeSet** — no MultiSenGE / MultiSenNA.

### 6.2 EarthDial Table 5 — Ship Dataset (SAR), referred object detection

Numbers are **mAP@0.5-style** columns (size / count splits), not example F1.

| Model | Small | Medium | Large | Single | Multiple |
| ----- | ----: | -----: | ----: | -----: | -------: |
| GPT-4o | 0.70 | 0.90 | 3.20 | 1.20 | 0 |
| **EarthDial** | **12.14** | **26.02** | **35.56** | **26.03** | **6.06** |

### 6.3 EarthDial Table 7 — region captioning on SAR (R-1 / R-L / METEOR)

| Dataset | Model | Rouge-1 | Rouge-L | Meteor |
| ------- | ----- | ------: | ------: | -----: |
| SAR-Ship | GPT-4o | 17.68 | 11.81 | 9.63 |
| SAR-Ship | GeoChat | 57.15 | 57.15 | 52.2 |
| SAR-Ship | **EarthDial** | **63.1** | **63.1** | **54.83** |
| SRSDD-v1.0 | GPT-4o | 7.49 | 7.24 | 7.07 |
| SRSDD-v1.0 | GeoChat | 63.72 | 63.72 | 57.31 |
| SRSDD-v1.0 | **EarthDial** | **68.8** | **68.8** | **62.45** |

### 6.4 EarthDial LULC / scene **classification** — mainly **RGB** and **S2** (not AI4LCC S1)

EarthDial’s land-cover / scene understanding for **classification** is built on **optical RGB** and **Sentinel-2 multispectral** datasets (BigEarthNet, LCZ, TreeSatAI, etc.). That is where their LULC numbers live — **not** on MultiSenGE/NA OCSGE with Sentinel-1.

**Do not equate** these accuracies with your S1 patch **example F1**. Different datasets, labels (CORINE / LCZ / scene names ≠ 14 OCSGE), and often single-label vs your multi-label set + dialogue.

#### 6.4a Table 4 — RGB / optical classification (Accuracy %)

| Model | AID | UCM | WHU-19 | **BigEarthNet (RGB)** | xBD Set-1 | FMoW |
| ----- | --: | --: | -----: | --------------------: | --------: | ----: |
| GPT-4o | 74.73 | 88.76 | 91.14 | 49.0 | 67.95 | 21.43 |
| InternVL-8B | 60.40 | 58.23 | 79.30 | 19.73 | 51.44 | 21.04 |
| GeoChat | 72.03 | 84.43 | 80.09 | 20.35 | 53.32 | 59.20 |
| **EarthDial** | **88.76** | **92.42** | **96.21** | **68.82** | **96.37** | **70.03** |

- **BigEarthNet RGB** = EarthDial’s main published **land-cover multi-label** optical number (**68.82%** Acc).  
- AID / UCM / WHU-19 = **aerial scene** classification (not French OCSGE).  
- xBD / FMoW = disaster / functional land-use style tasks — still **not** S1 OCSGE dialogue.

#### 6.4b Table 5 — multispectral / S2-family classification (Accuracy %)

| Task | Input | GPT-4o | **EarthDial** | Δ vs GPT-4o |
| ---- | ----- | -----: | ------------: | ----------: |
| **BigEarthNet (MS)** | Sentinel-2 multispectral | 49.0 | **69.94** | +20.94 |
| **LCZ42** | S2 local climate zones | 15.53 | **60.72** | +45.19 |
| **TreeSatAI (RGBI)** | RGB + NIR tree species | 16.73 | **56.61** | +39.88 |

| Dataset in EarthDial pipeline | Modality | Role |
| ----------------------------- | -------- | ---- |
| `BigEarthNet_FINAL_RGB` / BigEarthNet RGB | Optical RGB | Multi-label LC classify (Table 4) |
| `BigEarthNet_S2` / BigEarthNet MS | **S2** (up to 12 bands) | Multi-label LC classify (Table 5) |
| `LCZs_S2` | **S2** | Local climate zone classify |
| `TreeSatAI` | RGBI | Tree species classify |

**Takeaway for novelty talk:** EarthDial already “does LULC classification” — but on **RGB/S2 BigEarthNet-style** data. Your contribution is **S1 VH + official OCSGE + dialogue + GE→NA**, where EarthDial ZS collapses (**example F1 0.019**).

### 6.5 Your DataSET / AI4LCC numbers (same protocol family)

| Dataset | Setting | Test N | example F1 | Dial T1 | Dial T2 |
| ------- | ------- | -----: | ---------: | ------: | ------: |
| MultiSenGE | EarthDial ZS | 801 | **0.019** | 0 | 0 |
| MultiSenGE | LULCDial ~25% | 801 | **0.782** | ~0.10 | ~0.38 |
| MultiSenGE | LULCDial ~50% | 801 | **0.783** | ~0.12 | ~0.38 |
| MultiSenGE | LULCDial 100% | 801 | **0.799** | ~0.12 | ~0.37 |
| MultiSenNA | GE→NA transfer (no NA FT) | 12115 | **0.670** | 0.018 | 0.081 |

**One-line for professor:**  
> EarthDial’s LULC classification is **RGB/S2 BigEarthNet** (~**69%** Acc); its S1 metrics are ships/captions — not French OCSGE. On MultiSenGE S1 LULC dialogue, EarthDial ZS is near floor (**0.019**); after LULCDial-S1 FT → **~0.80**.

**Public search (Jul 2026):** no paper found that builds an instruction/dialogue bench on AI4LCC OCSGE or reports VLM results on MultiSenGE/NA as the main LULC chat task → supports **N1** (with “to the best of our knowledge”).

---

## 7. Side-by-side: their scores vs yours (for slides — with disclaimer)

| Work | Task | Input | Metric | Number | Comparable to you? |
| ---- | ---- | ----- | ------ | -----: | ------------------ |
| ISPRS 2022 U-Net-IRRG | Urban seg (Metz) | S2 IRRG | Weighted F1 | **0.74** | **No** |
| RS 2023 ConvLSTM+Inception | LULC seg (6-cls highlight) | S1+S2 multi-date | Weighted F1 | ~**0.90** | **No** |
| RSL 2025 FT on NA | LULC seg | S1+S2 multi-date | Acc / wF1 | **81.9% / 83.2%** | **No** (and they **train on NA**) |
| EarthDial CVPR 2025 Table 5 | SAR ship referred det. | SAR ships | mAP-style | e.g. Large **35.56** | **No** (ships ≠ OCSGE) |
| EarthDial Table 4 BigEarthNet | LC multi-label classify | **RGB** | Acc % | **68.82** | **No** (optical CORINE-style ≠ OCSGE S1) |
| EarthDial Table 5 BigEarthNet MS | LC multi-label classify | **S2 MS** | Acc % | **69.94** | **No** |
| EarthDial Table 5 LCZ42 | Climate-zone classify | **S2** | Acc % | **60.72** | **No** |
| EarthDial Table 5 TreeSatAI | Tree species classify | RGBI | Acc % | **56.61** | **No** |
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

> “MultiSenGE/NA are already used for **CNN semantic segmentation** — U-Net about **0.74** weighted F1 on urban Metz with S2, and later ConvLSTM S1+S2 papers report much higher pixel F1, and RSL 2025 fine-tunes CNNs **on** MultiSenNA. EarthDial’s LULC classification is mainly **RGB / S2 BigEarthNet** (~**69%** Acc), and its published S1 numbers are **ships/captions** — not OCSGE. On our MultiSenGE S1 LULC dialogue, EarthDial ZS is ~**0.02**, then LULCDial FT ~**0.80**.  
> I am **not** competing on that pixel or BigEarthNet leaderboard. I built the first **VLM instruction benchmark** on the same OCSGE labels with **S1-only** dialogue, raised EarthDial from **0.02 → 0.80** example F1 on GE val, and got **0.67** on MultiSenNA **without** NA training.  
> So the novelty is **interactive SAR LULC understanding and transfer**, complementary to optical/S2 classifiers and CNN mappers.”

---

## 10. References (quick)

1. Wenger et al. (2022) MultiSenGE dataset paper — ISPRS Annals.  
2. Wenger et al. (2023) ConvLSTM+Inception on MultiSenGE — Remote Sensing 15(1):151.  
3. Wenger et al. (2025) MultiSenGE ↔ MultiSenNA CNN fine-tuning — Remote Sensing Letters.  
4. AI4LCC portal — https://doi.theia.data-terra.org/ai4lcc/  
5. EarthDial (CVPR 2025) — LULC **classify** mainly **RGB / S2** (BigEarthNet Acc **68.82** / **69.94**; LCZ **60.72**; TreeSatAI **56.61**); S1 = ships / captions / QuakeSet; **not** AI4LCC OCSGE dialogue.  
6. Your metrics — `LULCDial-s1/data/baresoil_s1/metrics/v0.1/`

---

*Internal note: if you later need a fair numeric “vs CNN” column, train a simple multi-label CNN/MLP on the **same** `ai4lcc_val.jsonl` patch targets and report example F1 — do not reuse pixel weighted F1 from Metz.*
