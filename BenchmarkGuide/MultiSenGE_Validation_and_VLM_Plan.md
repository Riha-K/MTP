# MultiSenGE Validation + VLM Extension Plan

**(A) Validation** — improve MultiSenGE **pixel segmentation** results with a modern model under a fair protocol.  
**(B) Extension** — show MultiSenGE (and MultiSenNA) are useful for **VLM** LULC classify + dialogue + regional transfer.  
EarthDial / LULCDial-S1 = **tool for (B)**, not the claimed base contribution.

> **Active CNN path (locked):** Remote Sensing 2023 ConvLSTM+Inception-S1S2 (geographic tile split, 4-date S1+S2). Metz-only / single-date IRRG ideas below are **historical survey notes — not the active plan**.

---

## 2. Papers on MultiSenGE / AI4LCC (published so far)

Almost all public DL results are from the **same author group** (Wenger, Puissant, Forestier et al.). 

### 2.1 Timeline

Data download (not a paper): [THEIA AI4LCC](https://doi.theia.data-terra.org/ai4lcc/)


| Year            | Authors                                                                             | Full title                                                                                                                                          | Uses MultiSenGE?                  | Local PDF                                                          |
| --------------- | ----------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- | ------------------------------------------------------------------ |
| **2022**        | Romain Wenger, Anne Puissant, Jonathan Weber, Lhassane Idoumghar, Germain Forestier | U-Net feature fusion for multi-class semantic segmentation of urban fabrics from Sentinel-2 imagery: an application on Grand Est Region, France     | **No** (precursor / same region)  | `Wenger_etal_2022_UNet_FeatureFusion_IJRS_author.pdf`              |
| **2022**        | Romain Wenger, Anne Puissant, Jonathan Weber, Lhassane Idoumghar, Germain Forestier | MULTISENGE: A Multimodal and Multitemporal Benchmark Dataset for Land Use/Land Cover Remote Sensing Applications                                    | **Yes** (dataset + weak baseline) | `Wenger_etal_2022_MultiSenGE_ISPRS_Annals.pdf`                     |
| **2022 / 2023** | Romain Wenger, Anne Puissant, Jonathan Weber, Lhassane Idoumghar, Germain Forestier | Multimodal and Multitemporal Land Use/Land Cover Semantic Segmentation on Sentinel-1 and Sentinel-2 Imagery: An Application on a MultiSenGE Dataset | **Yes**                           | `Wenger_etal_2023_ConvLSTM_Inception_MultiSenGE_RemoteSensing.pdf` |
| **2023**        | Romain Wenger, Anne Puissant, Jonathan Weber, Lhassane Idoumghar, Germain Forestier | Exploring inference of a land use and land cover model trained on MultiSenGE dataset                                                                | **Yes**                           | `Wenger_etal_2023_JURSE_MultiSenGE_Inference.pdf`                  |
| **2025**        | Romain Wenger, Anne Puissant, Germain Forestier                                     | Multitemporal and Multimodal Cross-Region Fine-Tuning: MultiSenGE and MultiSenNA Applications                                                       | **Yes** (+ MultiSenNA)            | `Wenger_etal_2025_MultiSenNA_CrossRegion_RSL.pdf`                  |


---

## 3. What each paper did - and how they showed “improvement”

### P0 - ISPRS 2022 (MultiSenGE release) - **the baseline your professor often quotes**


| Item    | Detail                                                                    |
| ------- | ------------------------------------------------------------------------- |
| Task    | Pixel semantic segmentation                                               |
| Place   | **Metz / tile T31UGQ** (not all 8,157 patches)                            |
| Classes | Urban **1–5** + aggregated **other** → **6 classes**                      |
| Input   | **Single-date Sentinel-2 IRRG** (3 bands) — **no S1**, **no time series** |
| Model   | U-Net + VGG-16                                                            |
| Metric  | **Weighted F1**                                                           |
| Numbers | U-Net-IRRG **0.7364**; U-Net-Index **0.7214**                             |


**How they “show” results:** one table of weighted + per-class F1 on Metz. Explicitly say multimodal/multitemporal DL is future work.

**Takeaway for you:** This is the cleanest “same protocol” starting point, but it is **not** the strongest published MultiSenGE number anymore.

---

### P1 - Remote Sensing 2023 (ConvLSTM+Inception-S1S2) - **real MultiSenGE upgrade**


| Item                      | Detail                                                                                                      |
| ------------------------- | ----------------------------------------------------------------------------------------------------------- |
| What changed vs ISPRS     | Use **4-date** S2 (+ S1) stacks; **geographic** train/val/test by S2 tiles; 6-class **and** 10-class merges |
| Model                     | ConvLSTM (temporal) ± Inception (spectral) → **U-Net** (VGG-16)                                             |
| Ablations                 | S1-only, S2-only, S1+S2, +Inception                                                                         |
| How they show improvement | Ablation tables: Weighted F1 / Precision / Recall / Kappa; confusion matrices; qualitative maps             |


**Report:**

- **6-class:** ConvLSTM-S1S2 / +Inception reach **Weighted F1 ≈ 0.89–0.90** (e.g. 0.9018 / 0.8875 depending on method row).
- **10-class:** ConvLSTM+Inception-S1S2 best overall (**Acc ≈ 0.88**, Weighted F1 quoted around **0.64–0.89** family metrics in text; **Kappa 0.7945** best).
- Claim: discriminating the over-represented “other” / natural bag (**10-class** vs **6-class**) helps **urban fabric** F1.

---

### P2 - JURSE 2023 (cross-city inference)


| Item                              | Detail                                                                                                         |
| --------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| Setup                             | Train on MultiSenGE; **infer** on Toulouse, Dijon, Orléans, Lille, Rennes (manual digitization for eval)       |
| Classes                           | 5 UF + aggregated natural                                                                                      |
| How they show improvement / value | City-level weighted F1 **~0.69–0.81** (Dijon best **0.8087**; Lille lowest **0.6866**); per-class tables; maps |


**Takeaway:** CNN geographic transfer already explored. Your MultiSenNA **VLM** transfer is a **different task** (patch classify/dialogue vs pixel maps) - complementary, not a duplicate if you say so clearly.

---

### P3 -  RSL 2025 (MultiSenNA + fine-tuning)


| Item        | Detail                                                                                          |
| ----------- | ----------------------------------------------------------------------------------------------- |
| New data    | **MultiSenNA** (~12k patches), class **15 Beaches/Sand** added                                  |
| Model       | **Same** ConvLSTM+Inception-S1S2 (no new backbone)                                              |
| Idea        | Fine-tune GE↔NA; freeze encoder; reduce data size; report **F1 + carbon (kg) + time**           |
| Best quoted | MultiSenNA wF1 **83.17%** (FT from GE); MultiSenGE wF1 **86.47%** (FT from NA) in their Table 2 |


**Takeaway:** Their “transfer” is **CNN weight fine-tuning** and sustainability. Your LULCDial **zero-shot / no-NA-train** VLM MultiSenNA F1 **0.679** is another transfer story — report side-by-side as different modalities/tasks.

---

## 4. Research gaps (where you can still contribute)


| Gap                                    | Status                                                       | Your opportunity                                                                                                      |
| -------------------------------------- | ------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------- |
| **Modern architectures on MultiSenGE** | Literature still dominated by **U-Net / ConvLSTM+Inception** | Re-benchmark with **U-TAE, SegFormer, UPerNet+ConvNeXt/Swin, optional EO foundation models** on a **frozen protocol** |
| **Independent third-party validation** | Almost all papers = same group                               | Your lab becomes an external validation of AI4LCC                                                                     |
| **Fair public leaderboard**            | Splits differ (Metz vs tile geo-split vs MD5 VLM)            | Publish **one locked protocol** + code + numbers                                                                      |
| **Full 14-class pixel maps**           | Mostly **6 or 10** merged classes                            | Harder full-14 table (optional stretch)                                                                               |
| **SAR-only modern nets**               | Underplayed vs S1+S2                                         | SAR-only ablation (aligns with your S1 VLM story)                                                                     |
| **VLM / dialogue on OCSGE**            | **No published MultiSenGE VLM**                              | Your SAR-LC-Bench + LULCDial fills this (**extension**)                                                               |
| **VLM regional transfer**              | CNN FT exists (2025); VLM no-retrain less studied            | MultiSenNA with LULCDial (**already done**)                                                                           |


---

## 5. How to show improvement (active CNN plan)

**Only active validation path:** Remote Sensing 2023 protocol.

- Match RS 2023 as closely as possible: **4 dates** (Jul/Aug/Sep/Nov), S1+S2, **geographic tile split**, 6- and/or 10-class merges  
- Reproduce ConvLSTM+Inception → **VGG-16 U-Net** Weighted F1 / Kappa  
- Then same protocol + advanced model (**U-TAE** / SegFormer-temporal / etc.)  
- Success: equal or beat their scores under the **same split**

*(Historical note: ISPRS Metz IRRG U-Net ~0.74 was considered, then dropped as too weak a baseline.)*

---

## 6. Latest models to use (after replicate)


| Priority | Model                                                     | Why                                           |
| -------- | --------------------------------------------------------- | --------------------------------------------- |
| 0        | **ConvLSTM+Inception → VGG-16 U-Net**                     | Paper replicate (required first)              |
| 1        | **U-TAE** (Garnot & Landrieu)                             | Standard SITS temporal attention for Sentinel |
| 2        | **SegFormer / Swin-UNet** on stacked dates or late fusion | Modern transformer spatial encoder            |
| Optional | **Prithvi-EO / TerraTorch** fine-tune                     | “2025–26 foundation” row (optional, heavier)  |


---

## 7. Wholesome project plan (Pillar A + Pillar B)

```text
                    MultiSenGE (ISPRS 2022) = BASE
                              |
          +-------------------+-------------------+
          |                                       |
   PILLAR A — VALIDATION                    PILLAR B — EXTENSION
   Pixel LULC segmentation                  VLM usability of same taxonomy
          |                                       |
   Multitemp S1+S2 (RS 2023)                SAR-LC-Bench (classify + 2-turn)
   replicate ConvLSTM+Inception             LULCDial-S1 (EarthDial FT)
          |                                       |
   Advanced model same protocol             MultiSenNA regional transfer
                                            (no NA training — already F1 0.679)
```

Live phase checklist: root [`ROADMAP.md`](../ROADMAP.md). VLM work is **B** (done). Missing for professor is **CNN replicate + advanced model**.

---

## 8. What to do **next** (locked with professor)

1. RS-2023 ConvLSTM+Inception-S1S2 only (Metz IRRG path dropped).  
2. S2 + ground_reference already on disk — next is protocol fidelity + PARAM train.  
3. Email authors for model code; reimplement continues in `multisenge_seg/`.  
4. Train replicate on PARAM → advanced model under same protocol.  
5. Keep LULCDial numbers frozen for Extension § — separate tables from CNN.

---

## 9. How to present to professor (talking points)

1. **Base = MultiSenGE.** EarthDial is only the VLM we adapted.
2. Published line: U-Net 0.74 (2022) → ConvLSTM+Inception ~0.90 (2023) → MultiSenNA FT (2025). Gap = **independent re-benchmark + modern model**.
3. We **validate** against the strongest same-team method (RS 2023), not the weak Metz IRRG row alone.
4. We also **extended** MultiSenGE to **VLM classify+dialogue** and showed **MultiSenNA** generalization without NA fine-tune (separate metrics).
5. Outcome = **validation + extension**, not extension alone.

---

## 10. Risks and honesty


| Risk                                            | Mitigation                                                                  |
| ----------------------------------------------- | --------------------------------------------------------------------------- |
| Exact Metz split hard to reconstruct from paper | Document assumptions; or ask authors / use RS2023 tile geo-split as primary |
| Beating 0.74 is “too easy” after 2023 paper     | Always compare to ConvLSTM+Inception (RS 2023) as the main baseline |
| S2 download / storage heavy                     | Start V1 (3 bands, one tile)                                                |
| Mixing VLM and CNN scores                       | Separate sections and tables forever                                        |


---

## 11. Bottom line

- **Published MultiSenGE DL work (2022–2025)** mostly upgrades **data usage** (time + S1) and **transfer/carbon**, still on a **ConvLSTM+U-Net** backbone family.  
- Professor wants **validation**: modern CNN/transformer on a **locked MultiSenGE protocol** with clear metric gains.  
- Your VLM + MultiSenNA work is the **extension** that makes the thesis broader — keep it, but **add Pillar A next**.  
- **Next build step:** Metz IRRG 6-class reproduce + SegFormer/DeepLab beat **0.7364**.

---

*Survey is best-effort from public DOIs/PDFs as of 2026-08-10. Re-check Scholar/OpenAlex before paper submission for any new 2025–2026 MultiSenGE citations.*