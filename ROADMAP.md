# MultiSenGE validation + LULCDial extension - Project roadmap

> **History:** `[log.md](log.md)` · **VLM commands:** `[LULCDial-s1/RUNBOOK.md](LULCDial-s1/RUNBOOK.md)` · **Survey:** `[BenchmarkGuide/MultiSenGE_Validation_and_VLM_Plan.md](BenchmarkGuide/MultiSenGE_Validation_and_VLM_Plan.md)`

---

## 1. One project, two pillars


| Pillar             | Goal                                                                                                                                 | Status                                |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------- |
| **A — Validation** | Replicate Remote Sensing 2023 ConvLSTM+Inception-S1S2 on MultiSenGE, then beat/match with a modern model under the **same** protocol | **NEXT**                              |
| **B — Extension**  | SAR-LC-Bench + LULCDial-S1 (S1 VH · 14-class OCSGE · dialogue) + MultiSenNA transfer                                                 | **Core done** (optional polish later) |


---

## 2. Validation protocol (CNN)

**Paper:** Wenger et al., *Multimodal and Multitemporal Land Use/Land Cover Semantic Segmentation on Sentinel-1 and Sentinel-2 Imagery: An Application on a MultiSenGE Dataset*, Remote Sensing 2023.  
**Code:** `multisenge_seg/` · **Protocol:** `multisenge_seg/PROTOCOL.md`


| Item    | Setting (target)                                                    |
| ------- | ------------------------------------------------------------------- |
| Sensors | Multitemporal **S1 + S2** (≈ 4 dates: Jul/Aug/Sep/Nov, ≥17-day gap) |
| Model   | **ConvLSTM + naive Inception → U-Net (VGG-16)**                     |
| Classes | **6** and/or **10** (merged; urban focus)                           |
| Split   | **Geographic** by Sentinel-2 tiles (as in paper Fig. 4)             |
| Metric  | Weighted F1 / per-class F1 / Kappa (pixel segmentation)             |


*(Earlier Metz-only IRRG baseline idea was dropped — this RS-2023 setup is the only CNN track.)*

---

## 3. Data sizes 


| Archive                | Approx size | Need for CNN validation?                        |
| ---------------------- | ----------- | ----------------------------------------------- |
| `s1.tgz`               | **~110 GB** | Yes (already used for LULCDial)                 |
| `s2.tgz`               | **~80 GB**  | Yes ✅ on disk (~88 GB extracted)               |
| `ground_reference.tgz` | **~25 MB**  | Yes ✅                                           |
| `labels.tgz`           | ~few MB     | Yes                                              |


URLs: [THEIA AI4LCC](https://doi.theia.data-terra.org/ai4lcc/) · Unistra S3 links in Zenodo MultiSenGE page.

---

## 4. Where to run what


| Machine                | Do this                                                                                                                                  |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| **Your laptop**        | Protocol, index, model/code, docs, email authors; no full train                                                                          |
| **PARAM GPU**          | Train replicate + advanced model; long jobs                                                                                              |

**Transfer:** lab box [`LAB_GPU_TRANSFER.md`](multisenge_seg/LAB_GPU_TRANSFER.md) · PARAM [`PARAM_TRANSFER.md`](multisenge_seg/PARAM_TRANSFER.md)


**Do not** try full training on the laptop.

---

## 5. Phase checklist


| Phase   | Goal                                                             | Machine        | Status                                      |
| ------- | ---------------------------------------------------------------- | -------------- | ------------------------------------------- |
| **A0**  | Freeze protocol (tiles, months, 6/10-class map, metrics)         | Laptop         | ✅ `multisenge_seg/PROTOCOL.md`              |
| **A0b** | Email authors for model code / weights                           | Laptop         | Pending (`EMAIL_AUTHORS_CODE_REQUEST.md`)   |
| **A1**  | Download S2 + GR                                                 | Laptop         | ✅ S2 ~88 GB · GR 8157                       |
| **A2**  | Build 4-date train/val/test index (+ optional cache)             | Laptop         | ✅ ~5890 after adding tile **32ULV**; S1 by month match |
| **A3**  | Reimplement ConvLSTM+Inception → VGG-16 U-Net in PyTorch         | Laptop → PARAM | ✅ VGG-16 U-Net + train (aug / ReduceLR / EarlyStop / norm) |
| **A4**  | Train replicate; report Weighted F1 vs paper                     | PARAM          | **In progress** — job 96769 (`sbatch train.sbatch`) |
| **A5**  | Same protocol + advanced model (e.g. U-TAE / SegFormer-temporal) | PARAM          | Pending                                     |
| **B**   | VLM numbers frozen for Extension §                               | —              | Done                                        |


---

## 6. Pillar B (done) — pointers


| Item                        | Where                                                      |
| --------------------------- | ---------------------------------------------------------- |
| Commands / PARAM            | `[LULCDial-s1/RUNBOOK.md](LULCDial-s1/RUNBOOK.md)`         |
| Older VLM-only plan archive | `[LULCDial-s1/ROADMAP_VLM.md](LULCDial-s1/ROADMAP_VLM.md)` |
| Metrics                     | `LULCDial-s1/data/.../metrics/v0.1/`                       |
| Bench draft                 | `sar_lc_bench_v0.1/`                                       |
| Progress report             | `writeup/PHASE1_PROGRESS_REPORT.md`                        |


GE test F1 **0.812** · MultiSenNA **0.679** (post-radiometry-fix).

---

## 7. Doc map


| File                                                   | Role                               |
| ------------------------------------------------------ | ---------------------------------- |
| `ROADMAP.md` (this)                                    | Whole-project plan — Pillar A next |
| `LULCDial-s1/RUNBOOK.md`                               | LULCDial/VLM copy-paste commands   |
| `log.md`                                               | Chronological changes              |
| `BenchmarkGuide/MultiSenGE_Validation_and_VLM_Plan.md` | Paper survey + gaps                |
| `BenchmarkGuide/*.pdf`                                 | MultiSenGE paper PDFs              |


*Updated 2026-08-11 — single CNN track (RS-2023); VGG-16 U-Net in code.*
