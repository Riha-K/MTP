# CVPR 2026 - modality-specific S1/S2 encoders + LULC relevance

**Deep study notes (Option A SAR-DINO / Option B TESSERA):** [`STUDY_NOTES_SAR_DINO_and_TESSERA.md`](STUDY_NOTES_SAR_DINO_and_TESSERA.md)

[CVPR 2026 Open Access](https://openaccess.thecvf.com/CVPR2026)

**Target next architecture:**

```
S1 → SAR-specific encoder → S1 L-TAE      ─┐
                                           ├─ gate / fusion → decoder
S2 → optical-specific encoder → S2 L-TAE  ─┘
```

---

## 1. Is each downloaded paper LULC?


| File                             | Task                                      | LULC / land-cover?                       | Datasets (main)                                                    | Notes for us                                                                                                      |
| -------------------------------- | ----------------------------------------- | ---------------------------------------- | ------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| `Wei_MM-OVSeg_…`                 | Open-vocab **segmentation** (optical+SAR) | **Partial** (City/Forest/Farmland/…)     | PIE-RGB-SAR, DDHR                                                  | Encoder asymmetry (DINO_rgb ≠ DINO_sar). **Not** MultiSenGE / urban 6c taxonomy.                                  |
| `Feng_TESSERA_…`                 | Pixel embeddings / retrieval / transfer   | **Related** (can transfer to land tasks) | Global S1/S2 time series                                           | Dual optical/radar branches. Not a fixed LULC taxonomy paper.                                                     |
| `Houdre_RAMEN_…`                 | EO foundation + **seg** benchmarks        | **Yes (among tasks)**                    | PASTIS (crop parcels), FLAIR-HUB land cover, Sen1Floods11, PANGAEA | S2 vs S2+S1 ablations; Late **LTAE**. Closest FM↔our stack.                                                       |
| `Houdre_RAMEN_supplemental_…`    | RAMEN supp                                | **Yes**                                  | PASTIS + Late LTAE tables                                          | Cite for LTAE fusion numbers.                                                                                     |
| `Wu_SkySense-VITA_…`             | In-context multimodal **seg**             | **Yes (broad RS classes)**               | Sky-VT-300k (176 cats, optical+SAR)                                | Align then fuse; not MultiSenGE-specific.                                                                         |
| `Shu_TerraScope_…`               | VQA / pixel reasoning                     | **Partial** (land questions)             | Terra-CoT, TerraScope-Bench                                        | Adaptive optical vs SAR selection ≈ gating idea.                                                                  |
| `Herzog_OlmoEarth_…`             | Multimodal EO FM                          | **Related** (WorldCover etc. in mix)     | S1, S2, Landsat + maps                                             | Foundation, not MultiSenGE LULC.                                                                                  |
| `Chen_SpectralMoE_…`             | Domain-gen **seg** (spectral shift)       | **Partial**                              | Hyperspectral / multispectral / FLAIR-related                      | Dual-gated MoE — fusion idea, not S1/S2 LULC.                                                                     |
| `Zhang_FUSAR-GPT_…`              | SAR VLM                                   | **No** (caption / VQA on SAR)            | FUSAR-GEOVL                                                        | Supports “SAR needs own stack.”                                                                                   |
| `Liu_CRFT_…`                     | Optical–SAR **registration**              | **No**                                   | Registration pairs                                                 | Alignment only.                                                                                                   |
| `Zhao_MOS_…`                     | Ship **ReID** optical↔SAR                 | **No**                                   | Ship ReID                                                          | Modality gap, wrong task.                                                                                         |
| `Weitzel_50Cities_…` (**CVPRW**) | Multimodal **urban land-cover seg**       | **Yes — LULC**                           | 50 Cities S1+S2                                                    | **Explicit S1-only vs S2-only vs both** (U-Net / SegFormer). Best “same model family, modality matters” citation. |
| `Forgaard_THOR_…` (**CVPRW**)    | EO foundation                             | **Yes (PASTIS etc.)**                    | PASTIS, CropMap, PANGAEA                                           | Land / crop segmentation benchmarks.                                                                              |


**Summary:** First batch was mostly **architecture / FM / fusion**, not MultiSenGE-style urban LULC. Added **50 Cities** + **THOR** for LULC-facing evidence. Still **no CVPR 2026 MultiSenGE clone** — your RS 2023 MultiSenGE paper remains the fair LULC baseline.

---



## 2. Papers that work on **S1-only** or **S2-only** (or ablate them)


| Paper                                        | What they show                                                                                                                        | Why it helps your argument                                                                          |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| **50 Cities** (CVPRW 2026)                   | Same seg heads on **S2-only ≫ S1-only** (e.g. Asia test: U-Net S2 mIoU **56.4** vs S1 **42.1**; SegFormer S2 **47.2** vs S1 **20.5**) | Same architecture, modality changes performance a lot → SAR needs care / better fusion.             |
| **RAMEN**                                    | Tab. multimodal: **S2** vs **S2+S1** (e.g. Sen1Floods11 89.96 → 91.20; PASTIS 40.99 → 44.25)                                          | Adding S1 helps **when fusion is right**; also implies S1 alone is not the main driver.             |
| **Our own MultiSenGE runs**                  | U-TAE S1-only / S2-only / both (see `RESULTS_BOARD.md`)                                                                               | Direct thesis evidence under identical U-TAE.                                                       |
| **MM-OVSeg**                                 | Trains a **SAR-specific** DINO (not “S2 encoder with 2 channels”)                                                                     | Architecture answer to “S1 needs its own encoder.”                                                  |
| **FUSAR-GPT**                                | Entire stack built for **SAR-only** VLM                                                                                               | Extreme case of modality-specific design.                                                           |
| **ORSATR-X** (CVPR 2026, optical RS objects) | Optical-only foundation for RS objects                                                                                                | Optical-specific path (cite if discussing S2 encoder family). PDF not stored here unless requested. |


There are few CVPR 2026 papers that are **pure S1-only LULC** or **pure S2-only LULC** end-to-end products; the strong pattern is: **ablate S1 vs S2 vs both**, or use **separate encoders**.

---



## 3. Priority for *your* next model (sir’s diagram)


| Pri | Read                       | Takeaway                                        |
| --- | -------------------------- | ----------------------------------------------- |
| P0  | MM-OVSeg                   | Different encoder for SAR vs optical, then fuse |
| P0  | TESSERA                    | Dual-branch S1/S2 temporal stacks               |
| P0  | **50 Cities**              | LULC dataset + S1-only ≪ S2-only numbers        |
| P1  | RAMEN (+ supp)             | PASTIS LULC/crop + Late LTAE + S2 vs S2+S1      |
| P1  | SkySense-VITA / TerraScope | Align / adaptive gate ideas                     |
| P2  | THOR                       | Extra LULC/PASTIS FM context                    |
| P3  | FUSAR / CRFT / MOS         | Supporting only                                 |


---



## 4. Practical recommendation

1. Keep reporting **S1-only / S2-only / both** (you already have) — that *is* the modality-specific empirical story.
2. For novelty: change MA-UTAE so **S1 encoder ≠ S2 encoder** (widths / SAR init / separate DINO-style stem), keep **separate L-TAEs** + gate.
3. Cite **MM-OVSeg + TESSERA** for architecture; **50 Cities + RAMEN + your MultiSenGE numbers** for LULC / modality gap.

---



## Search notes

CVF `SAR` search is flaky; use `Optical`, `Earth Observation`, EarthVision workshop. Index: [https://openaccess.thecvf.com/CVPR2026](https://openaccess.thecvf.com/CVPR2026)