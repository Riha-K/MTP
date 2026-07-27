# Research Brief: MultiSenGE (CNN) & EarthDial (SAR VLM)

> **Author lens:** Senior research scientist perspective — written as if encountering both papers fresh.  
> **Scope:** Two **independent** research lines. No comparison between them.  
> **Your modality focus:** **SAR (Sentinel-1)** is called out explicitly in Part B and in MultiSenGE SAR-only experiments (Part A).

---

# PART A — MultiSenGE / AI4LCC (2022): Can modern CNNs improve the baseline?

## A1. What the paper actually did

**Paper:** Wenger et al., ISPRS Annals V-3-2022 — *MULTISENGE: A Multimodal and Multitemporal Benchmark Dataset for LULC Remote Sensing Applications*  
**DOI:** [10.5194/isprs-annals-V-3-2022-635-2022](https://doi.org/10.5194/isprs-annals-V-3-2022-635-2022)

MultiSenGE is **not** a model paper. It is a **dataset + weak baseline** paper:


| What they built | Detail                                                                         |
| --------------- | ------------------------------------------------------------------------------ |
| Geography       | Grand-Est, France — 57,433 km²                                                 |
| Patches         | **8,157** non-overlapping **256×256 @ 10 m** triplets                          |
| Modalities      | Sentinel-1 (VV+VH GRD), Sentinel-2 L2A (10 bands), **ground_reference** raster |
| Labels          | **14 OCSGE classes** (5 urban + 9 natural)                                     |
| Temporal depth  | **72,033** S2 dates · **1,012,227** S1 dates (full 2020 stacks per patch)      |
| Tasks           | Semantic segmentation + scene classification (metadata in JSON)                |


**Critical sentence from the paper itself:**

> *"Others semantic segmentation deep learning methods using the **multimodal and multitemporal** dataset are a **part of an ongoing PhD research**."*

So the authors **never finished** the main scientific promise of the dataset. The published baseline is deliberately minimal.

---



## A2. The only baseline they published (what you would need to beat)

They ran **one** experiment — not on the full 8,157 patches:


| Setting   | Value                                                                        |
| --------- | ---------------------------------------------------------------------------- |
| Task      | Urban semantic segmentation (classes 1–5 + aggregated “other” = 6 classes)   |
| Region    | **Metz**, tile **T31UGQ** only                                               |
| Split     | 80% train / 20% val + **spatially separate test zone** (Saraiva et al. 2020) |
| Input     | **Single-date Sentinel-2 IRRG** (3 bands) — **NOT SAR, NOT multitemporal**   |
| Models    | **U-Net + VGG-16** (ImageNet pretrained vs random init with extra indices)   |
| Loss      | Weighted categorical cross-entropy                                           |
| Optimizer | Adam, LR 0.0001, batch 8                                                     |


**Results (Metz test):**


| Model                                             | Weighted F1 |
| ------------------------------------------------- | ----------- |
| U-Net-IRRG (VGG-16, ImageNet)                     | **0.7364**  |
| U-Net-Index (IRRG + NDVI/NDBI/eNDVI, random init) | 0.7214      |


Per-class urban F1 is much lower (e.g. dense built-up ~0.52, roads ~0.59). The paper’s own Figure 6 shows fragmented predictions.

**Honest conclusion:** The 2022 baseline is **optical, single-date, one tile, urban-only**. It does **not** use:

- Sentinel-1 at all
- Multitemporal stacks (the dataset’s main selling point)
- Full 14-class mapping on all 8,157 patches
- Any architecture from 2023–2026

**Yes — there is real room to improve.** But only if you follow a **rigorous, same-task protocol**.

---

## A3. Is there a publishable chance? 

### When the answer is YES

You can publish if you treat it as a **systematic re-benchmark paper**, not “we tried a bigger network.”

Publishable claims look like:

1. **"We complete what MultiSenGE deferred: multitemporal + multimodal segmentation on AI4LCC."**
2. **"Under identical patch size, labels, and metrics, modern encoders beat the 2022 U-Net baseline by X points."**
3. **"SAR-only / SAR+S2 fusion reduces confusion on classes X, Y (cloud-robust mapping)."**
4. **"Temporal modeling on the 2020 stack beats single-date optical — validating the dataset design."**

These are **legitimate** because the original authors explicitly left multitemporal/multimodal DL to future work.

### When the answer is NO (reviewers will reject)

- Train on a different split than you report
- Mix patch-level classification with pixel segmentation metrics
- Use only 801 or 2497 patches without stating it’s a subset
- Compare your 2026 model to their 0.7364 F1 while using **SAR** and they used **S2 IRRG**
- Claim “state of the art” without running **DeepLabV3+ / SegFormer / U-TAE** as sanity checks

---



## A4. Full-proof experimental design (same spirit as MultiSenGE, upgraded models)

This is the protocol I would require before approving a paper internally.

### Step 1 — Lock the task (pick ONE primary task)


| Task ID | Description                                           | Same as paper?                              |
| ------- | ----------------------------------------------------- | ------------------------------------------- |
| **T1**  | Urban 6-class seg on Metz T31UGQ, single-date S2 IRRG | ✅ Exact replication + upgrade               |
| **T2**  | Full **14-class** seg on **all 8,157** patches        | Extension (stronger contribution)           |
| **T3**  | **SAR-only** (VV+VH) 14-class seg                     | **Your SAR focus** — not in 2022 paper      |
| **T4**  | **Multitemporal S1+S2** 14-class seg                  | **Dataset’s core promise** — highest impact |


**Recommendation:** Lead with **T4** (multitemporal S1+S2) as main result; include **T1** as “we reproduce and surpass the original baseline under their exact setting.”

### Step 2 — Lock the data protocol


| Rule             | Specification                                                                                |
| ---------------- | -------------------------------------------------------------------------------------------- |
| Patch size       | 256×256 @ 10 m                                                                               |
| Target           | `ground_reference` raster (14 classes)                                                       |
| Train/val/test   | **Spatial split** — never random pixel shuffle across geography                              |
| S1 preprocessing | Same as AI4LCC: GRD, VV+VH stacked, CNES-style speckle filtering (document if you reprocess) |
| S2 preprocessing | L2A, 10 m bands resampled; document cloud handling for temporal stacks                       |
| Class imbalance  | Weighted CE or focal loss — report which                                                     |
| Metrics          | **Weighted F1**, mean IoU, per-class F1 (same family as paper)                               |


Use MD5 or official split files if you adopt AI4LCC community splits; **document seed and patch lists** in supplementary material.

### Step 3 — Model ladder (2022 → 2026)

Run all of these on **the same splits**:


| Tier              | Architecture                                  | Role in paper                                                                                                                                 |
| ----------------- | --------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| **B0**            | U-Net + VGG-16, S2 IRRG single-date           | **Replicate** Wenger 2022 (Metz T1)                                                                                                           |
| **B1**            | U-Net + **ResNet-50** or EfficientNet-B4      | Modern CNN encoder, same U-Net head                                                                                                           |
| **B2**            | **DeepLabV3+** (ResNet-50)                    | Strong CNN baseline used in RS reviews (2024–2026)                                                                                            |
| **B3**            | **Attention U-Net** or **Swin-UNet**          | CNN + attention / hierarchical context                                                                                                        |
| **B4**            | **U-TAE** (temporal attention on SITS)        | ICCV 2021 — standard for **multitemporal** S2 stacks                                                                                          |
| **B5**            | **3D U-Net** or **ConvLSTM-U-Net**            | Temporal baseline for S1+S2 volume                                                                                                            |
| **B6**            | **LULC-Former / SpecSAR-Former** style fusion | 2024–2025 S1+S2 transformers ([LULC-Former](https://doi.org/10.1109/jstars.2025.3641788), [SpecSAR-Former](https://arxiv.org/abs/2410.03962)) |
| **B7 (optional)** | **Prithvi-EO-2.0** fine-tune (TerraTorch)     | Foundation-model line — not CNN but strong “2026 SOTA” anchor                                                                                 |


**For SAR-only (T3):** B0–B3 with **2-channel input (VV+VH)**; add Swin-UNet on spatiotemporal S1 ([Russo et al. 2025](https://arxiv.org/abs/2503.07230) shows seasonal S1 stacks beat plain 2D CNNs).

### Step 4 — Ablations reviewers expect


| Ablation                       | Question answered                |
| ------------------------------ | -------------------------------- |
| Single-date vs full 2020 stack | Was multitemporal data worth it? |
| S2-only vs S1-only vs S1+S2    | Does SAR help which classes?     |
| 2 dates vs 6 vs 12 vs all      | Temporal depth curve             |
| ImageNet pretrain vs random    | Same question as 2022 paper      |
| Loss weighting on rare classes | Urban subclasses still hard?     |




### Step 5 — Statistical rigor

- **≥3 seeds** for main table (mean ± std)
- Significance test on weighted F1 (paired bootstrap over patches)
- **Qualitative maps** on Metz + one rural tile + one failure case
- **Compute budget** table (GPU hours, carbon — Forestier 2025 MultiSenNA paper sets precedent)



### Step 6 — What “improvement” means (realistic targets)


| Setting                    | 2022 reference | Reasonable 2026 target                   | Ambitious                                    |
| -------------------------- | -------------- | ---------------------------------------- | -------------------------------------------- |
| Metz urban, S2 single-date | F1 **0.736**   | **0.78–0.82** with DeepLabV3+/Swin-UNet  | >0.85 if heavy tuning                        |
| Full 14-class, all patches | *Not reported* | Establish **new** baseline row           | —                                            |
| S1+S2 multitemporal        | *Not reported* | **+5–15 IoU points** over S2 single-date | Publishable story even if absolute F1 < 0.90 |


You do **not** need miracle numbers. You need **controlled gains + ablations** that explain **why** (temporal, SAR texture, fusion).

---



## A5. How to write and publish (venue map)


| Contribution type                       | Target venue                                         | Title pattern                                                                                |
| --------------------------------------- | ---------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| Full re-benchmark + multitemporal S1/S2 | **ISPRS J&P RS**, **Remote Sensing (MDPI)**, **RSL** | *"Revisiting MultiSenGE: Multitemporal Sentinel-1/2 Segmentation with Modern Deep Networks"* |
| SAR-only focus                          | **IEEE JSTARS**, **ISPRS Journal**                   | *"Sentinel-1 VH/VV Semantic Segmentation on AI4LCC: Beyond Single-Date Optical Baselines"*   |
| Short replication + uplift              | **ISPRS Annals / IGARSS**                            | *"Updating the MultiSenGE U-Net Baseline with DeepLabV3+ and Temporal Fusion"*               |
| Cross-region (MultiSenGE → MultiSenNA)  | **Remote Sensing Letters**                           | Extend Forestier 2025 with your best architecture                                            |


**Paper skeleton (8 pages):**

1. Introduction — MultiSenGE contribution + **explicit gap** (no multitemporal DL results in 2022)
2. Dataset & protocol — patch, OCSGE, splits (reproducible)
3. Methods — model ladder B0–B6
4. Experiments — main table + ablations
5. Discussion — which classes benefit from SAR/time; failure modes (speckle, mixed pixels)
6. Conclusion — new reference numbers for AI4LCC community

**Artifacts to release (mandatory for credibility):**

- Split JSON (patch IDs)
- Training config YAML
- Checkpoint for B0 replication + best model
- Prediction GeoTIFFs on test set for qualitative review

---

## A6. Related work to cite (mapping line, not VLM)


| Paper                                                        | Why cite                             |
| ------------------------------------------------------------ | ------------------------------------ |
| Wenger et al. 2022 ISPRS                                     | Original MultiSenGE                  |
| Wenger et al. 2022 Remote Sensing                            | Extended baseline details            |
| Forestier et al. 2025 MultiSenNA transfer                    | Latest AI4LCC collection follow-up   |
| Garnot et al. 2021 **U-TAE**                                 | Temporal segmentation standard       |
| Ruivo et al. 2021 Wide-area **S1** seg (7 CNN architectures) | SAR segmentation benchmarks          |
| Russo et al. 2025 **Swin-UNet + S1 seasonal stacks**         | Modern SAR temporal LC               |
| LULC-Former / SpecSAR-Former 2024–2025                       | S1+S2 fusion transformers            |
| Prithvi-EO-2.0 2024                                          | Foundation model comparison point    |
| 3D U-Net crop mapping (RS 2024)                              | Multimodal temporal fusion precedent |


---

## A7. Bottom line — Part A


| Question                                    | Answer                                                                                                                                 |
| ------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| Can latest CNNs improve MultiSenGE results? | **Yes**, with high confidence                                                                                                          |
| Same method as MultiSenGE?                  | Same **task family** (semantic seg on `ground_reference`), **not** same single U-Net — you upgrade architecture while keeping protocol |
| Publishable?                                | **Yes** if you re-benchmark systematically and report multitemporal/SAR results the 2022 paper never published                         |
| Full-proof?                                 | Follow **Section A4** — locked splits, model ladder, ablations, released patch lists                                                   |


---

---



# PART B — EarthDial (CVPR 2025): Gaps, improvements, SAR-focused research

## B1. What EarthDial is ?

**Paper:** Soni et al., CVPR 2025 — *EarthDial: Turning Multi-sensory Earth Observations to Interactive Dialogues*  
**arXiv:** [2412.15190](https://arxiv.org/abs/2412.15190) · **Code:** [github.com/hiyamdebary/EarthDial](https://github.com/hiyamdebary/EarthDial)

EarthDial is a **~4B parameter VLM** (InternViT-300M + MLP + Phi-3-mini) trained on **11.11M** remote-sensing instruction pairs. It targets **interactive** EO: classification, detection, grounding, captioning, VQA, **change detection**, disaster assessment, methane, UHI, LCZ, etc., across **RGB, SAR, Sentinel-2 MS, IR, NAIP, multi-temporal**.

It is a **foundation model + dataset + benchmark** paper — not a final solution.

---



## B2. Does the paper speak about improvement / limitations?

**Yes — explicitly in main text and supplementary.**

### What they claim is still hard (failure cases, Fig. A.4 supplement)


| Failure mode           | Example                                                             |
| ---------------------- | ------------------------------------------------------------------- |
| **Ambiguous scenes**   | “Medium tree” among many green trees                                |
| **Subtle change**      | Small bottom-right change missed in change detection                |
| **Similar classes**    | Office building vs multi-unit residential (temporal classification) |
| **Crowded SAR scenes** | Ship detection when training data is ship-heavy vs other objects    |
| **Complex texture**    | Model confuses visually similar regions                             |




### Architectural limitations (stated or implied)


| Limitation                       | Evidence                                                                                         |
| -------------------------------- | ------------------------------------------------------------------------------------------------ |
| **Simple spectral fusion**       | Non-RGB bands processed as **3-channel groups**, concatenated — not learned cross-band attention |
| **ViT frozen after Stage 1**     | SAR/MS seen mainly through adapter/MLP/LLM — encoder may underfit SAR speckle statistics         |
| **LoRA underperforms full FT**   | Supplement: ~201M LoRA params << full fine-tune on RS tasks                                      |
| **Max ~4 temporal frames**       | Scene classification capped at 4 images — far less than MultiSenGE-scale yearly stacks           |
| **No pixel segmentation output** | Boxes + text only — no mask generation for LULC                                                  |
| **Greedy decoding**              | Structured outputs (boxes) not constrained                                                       |




### Data limitations (Stage 3 / SAR)


| Issue                  | Detail                                                                                                   |
| ---------------------- | -------------------------------------------------------------------------------------------------------- |
| **RGB-heavy training** | Stage 1: **7.6M** pairs; Stage 3 (MS+SAR): **~2.5M** — SAR/MS underrepresented vs RGB                    |
| **SAR task mix**       | SAR instructions skew to **ships, earthquakes (QuakeSet), Satlas** — not land-cover taxonomy             |
| **Synthetic QA**       | Much data from **InternLM-XComposer2** auto-generation from labels — domain shift for specialized SAR LC |
| **Eval imbalance**     | Most of **44 benchmarks** are RGB-biased; few pure SAR LC eval sets                                      |


EarthDial **does not** claim SAR land-cover dialogue is solved. It claims a **unified starter model** for many EO tasks.

---



## B3. SAR in EarthDial — what it actually does today

From paper Table 5 and Stage 3 description:


| SAR use in EarthDial      | Content                                                     |
| ------------------------- | ----------------------------------------------------------- |
| **Ship detection**        | Ship dataset — referred detection, mAP                      |
| **Earthquake / disaster** | QuakeSet — bi-temporal SAR, earthquake occurred? magnitude? |
| **Change detection**      | Some SAR bi-temporal change sets in 44 benchmarks           |
| **Classification**        | BigEarthNet-MS, SoSAT-LCZ42, TreeSatAI — not OCSGE / AI4LCC |
| **Pretraining**           | Satlas + SkyScript include some SAR captions                |


**What SAR is NOT in EarthDial:**

- Official **14-class OCSGE** land-cover dialogue
- **VH/VV speckle-robust** fine-grained LULC reasoning at 10 m
- **Full-year S1 time series** QA (“how did backscatter evolve?”)
- **SAR-only** vs **S2** cross-modal analyst reasoning
- **Pixel-aligned** SAR explanations tied to reference maps

**For your SAR focus:** EarthDial proves SAR **can** enter a VLM pipeline — it does **not** solve SAR **land-cover science**.

---



## B4. Research gaps (SAR-first, EarthDial line)



### Gap 1 — **SAR land-cover instruction data scarcity**


| Fact                                                           | Implication                      |
| -------------------------------------------------------------- | -------------------------------- |
| SARChat-2M (2025) focuses on **ships, tanks, ports, aircraft** | Object-centric, not OCSGE/LULC   |
| EarthDial SAR Stage-3 ≈ disaster + ships + generic captions    | LC semantics under-trained       |
| MultiSenGE has **1M+ S1 patches** but **zero** dialogue labels | Opportunity: SAR LC instruct set |


**Research:** Build **SAR-LC-Instruct** — QA from patch labels + temporal metadata (rules + LLM verification). Not “compare to CNN” — **extend EarthDial’s missing SAR-LC modality**.

---



### Gap 2 — **Speckle + radiometry understanding in VLMs**

SAR VH/VV is **not natural image statistics**. EarthDial feeds SAR through a ViT pretrained on RGB-like pipelines (3-channel grouping).

**Research questions:**

- Does a **SAR-specific frontend** (speckle-aware normalization, Lee filter branch, complex-valued CNN stem) before ViT improve VH/VV LC QA?
- Can the VLM **verbalize radiometry** (“VH backscatter increased 3 dB — likely soil moisture”)?
- **Uncertainty:** When speckle dominates, can the model **abstain**?

**Publishable angle:** “Speckle-aware adaptation for RS-VLMs” — method paper, SAR benchmarks.

---



### Gap 3 — **Multitemporal SAR reasoning (>4 frames)**

EarthDial caps many temporal tasks at **4 frames**. MultiSenGE has **dozens to hundreds** of S1 acquisitions per patch in 2020.

**Research:**

- **Long-sequence SAR VLM:** hierarchical time summarization (monthly composites → dialogue)
- **Seasonal change QA:** “Was this field bare in March?” using S1 stack
- Compare **frame sampling policies** (uniform vs change-point vs attention)

**Related:** ChangeChat/DeltaVLM do **bi-temporal RGB change** — not **S1 yearly stacks + LC taxonomy**.

---



### Gap 4 — **No pixel grounding for LULC in VLMs**

EarthDial outputs **boxes**, not **segmentation masks**. For land-cover, analysts need **where** class X is.

**Research directions:**


| Approach                       | Description                                           |
| ------------------------------ | ----------------------------------------------------- |
| **Referring segmentation VLM** | “Highlight wetlands in this SAR patch” → output mask  |
| **Tool-use agent**             | VLM calls external seg model, explains map            |
| **Unified mask head**          | Extend EarthDial with mask decoder (SAM-style) for RS |


**SAR angle:** Pixel grounding on **VH** where optical is unavailable — disaster/cloud scenarios.

---



### Gap 5 — **Cross-sensor reasoning (SAR vs optical)**

EarthDial fuses modalities late in the LLM but rarely evaluates **“which sensor for which question?”**

**Research:**

- Benchmark: same geography, **S1-only vs S2-only vs fused** QA with **known answer from labels**
- Train model to **cite sensor evidence** (“S2 confirms vegetation; SAR suggests built structure”)
- **Cloud-gap scenario:** optical missing — SAR-only dialogue accuracy

---



### Gap 6 — **Evaluation gap for SAR LC**

EarthDial’s 44 datasets **under-represent** SAR land-cover. SARChat-Bench evaluates **detection/counting**, not **multi-label LC**.

**Research:** Define **SAR-LC-Bench**:


| Task                       | Metric                        |
| -------------------------- | ----------------------------- |
| Multi-label patch classify | Example F1 / micro-F1         |
| Temporal change describe   | Event F1 vs label deltas      |
| Class presence grounding   | Box/mask IoU                  |
| Robustness                 | Speckle noise / missing dates |


Release benchmark → citation magnet (same playbook as EarthDial’s 44 datasets).

---



### Gap 7 — **Data quality & hallucination**

Much EarthDial data is **LLM-generated** from sparse OSM-like labels (<3 labels filtered, but still synthetic).

**Research:**

- **Factuality scoring** against pixel labels (when available)
- **Human-in-the-loop** verification for SAR LC (expert annotators)
- **Contrastive training** to penalize answers inconsistent with VH statistics

---



## B5. Concrete SAR research projects (publishable, EarthDial-extended)

Ranked by clarity and feasibility.

### Project S1 — **SAR-LC-Instruct + fine-tune EarthDial** (Dataset + model)


| Item   | Spec                                                              |
| ------ | ----------------------------------------------------------------- |
| Input  | Sentinel-1 VH/VV 256×256 (dB clipped)                             |
| Labels | Multi-label OCSGE from existing LC datasets OR new SAR-LC patches |
| Tasks  | Classify, justify, temporal “what changed”, cloud-gap scenarios   |
| Model  | EarthDial_4B_MS Stage-3 continued FT                              |
| Eval   | Held-out patch F1 + human eval on 200 dialogues                   |
| Venue  | IGARSS, GRSL, ISPRS                                               |


**Novelty:** First **taxonomy-grounded SAR LC** instruction set — EarthDial never had this.

---



### Project S2 — **SAR temporal dialogue on image stacks**


| Item      | Spec                                                                  |
| --------- | --------------------------------------------------------------------- |
| Input     | 4–12 S1 dates per patch (sampled from 2020 stack)                     |
| Questions | Seasonality, harvest, flood backscatter drop                          |
| Baseline  | EarthDial 4-frame limit vs proposed **hierarchical temporal encoder** |
| Venue     | CVPR EarthVision / ECCV workshop                                      |


---



### Project S3 — **Speckle-aware SAR adapter for RS-VLMs**


| Item      | Spec                                                           |
| --------- | -------------------------------------------------------------- |
| Method    | Insert SAR adapter before InternViT; train on SARChat + SAR-LC |
| Ablations | Raw dB vs filtered vs 3-channel repeat                         |
| Metrics   | SARChat-Bench + your SAR-LC-Bench                              |
| Venue     | TGRS / CVPR                                                    |


---



### Project S4 — **SAR change analysis dialogue (SAR-ChangeChat)**

Extend ChangeChat/DeltaVLM idea to **SAR bi/multi-temporal**:

- Bi-temporal VH difference QA
- Earthquake / flood / land-clearing events
- Complement EarthDial QuakeSet with **LC-aware** change questions

**Venue:** Remote Sensing / ISPRS (DeltaVLM 2025 line is RGB-focused)

---



### Project S5 — **Open-vocabulary SAR LC (OV + SAR)**

Combine open-vocabulary segmentation (SegEarth-OV, GR-CoT 2026) with **SAR input**:

- Query: “areas with low VH backscatter and linear structure” → map
- Evaluate on LC datasets with text queries

**Venue:** CVPR / ICCV RS workshop

---



## B6. What EarthDial authors would likely do next (inferred roadmap)

Based on paper + supplement + 2025–2026 field movement:

1. **Better spectral/SAR fusion** — cross-attention over bands, not 3-channel chunking
2. **More SAR + MS instruction data** — balance Stage 1 RGB dominance
3. **Segmentation / mask output** — close gap vs mapping community
4. **Longer temporal context** — beyond 4 frames
5. **Smaller / edge models** — deployment on satellite downlink
6. **Agentic tool use** — STAC search, index computation, GIS APIs

**Your SAR thesis can sit in (1)(2)(4)(6)** without repeating their RGB benchmark suite.

---



## B7. Key papers for SAR + VLM line (2024–2026)


| Paper                                       | Relevance                                                  |
| ------------------------------------------- | ---------------------------------------------------------- |
| **EarthDial** CVPR 2025                     | Base model + training recipe                               |
| **SARChat-2M** 2025                         | SAR dialogue data + bench — **objects**, not LC            |
| **ChangeChat** 2024 / **DeltaVLM** 2025     | Change dialogue — RGB bi-temp                              |
| **EarthGPT / MMRS**                         | Earlier multi-sensor VLM                                   |
| **GeoChat**                                 | RGB grounding — no SAR LC                                  |
| **Prithvi-EO-2.0**                          | Segmentation FM — potential tool backend for VLM           |
| **Russo et al. 2025 Swin-UNet S1 temporal** | SAR LC mapping — **back-end**, not competitor to EarthDial |


---



## B8. Bottom line — Part B


| Question                            | Answer                                                                                                        |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| Does EarthDial discuss improvement? | **Yes** — failure cases, fusion limits, data imbalance; supplement discusses LoRA vs full FT                  |
| Biggest SAR gap?                    | **SAR land-cover dialogue + temporal stacks + pixel grounding** — not ship detection                          |
| What should you do?                 | **Extend EarthDial on SAR-LC**, not re-run RGB benchmarks                                                     |
| Publishable SAR research?           | **SAR-LC-Instruct**, **temporal SAR dialogue**, **speckle-aware adapters**, **SAR-ChangeChat**, **OV-SAR-LC** |
| Avoid                               | Claiming EarthDial solves SAR mapping; comparing unrelated CNN F1                                             |


---



## Appendix — Quick decision matrix (for you)


| Your goal                                          | Follow                                                                    |
| -------------------------------------------------- | ------------------------------------------------------------------------- |
| Senior wants **better MultiSenGE mapping numbers** | **Part A** — temporal S1/S2 segmentation, model ladder, ISPRS/RSL paper   |
| You want **SAR + language thesis**                 | **Part B** — SAR-LC-Instruct, extend EarthDial, new SAR-LC benchmark      |
| Both?                                              | **Two papers**, two tasks — do **not** merge metrics into one leaderboard |


---



## Appendix — Reading order (`BenchmarkGuide/papers/`)

Local PDFs under `BenchmarkGuide/papers/` (+ MultiSenGE in `BenchmarkGuide/AI4LCC/`).

### Must read first (this week)


| # | File | Why |
|---|------|-----|
| **1** | `papers/EarthDial_CVPR2025_2412.15190.pdf` | VLM base. Focus Stage 3 SAR, fusion limits, failure cases. |
| **2** | `AI4LCC/multiSenge_AI4LCC.pdf` | CNN track — U-Net baseline; what Multitemporal/multimodal DL left unfinished. |
| **3** | `papers/SARChat-2M_2025_2502.08168.pdf` | Closest SAR+language paper. Ships/objects — **not** LULC (your gap). |
| **4** | `papers/Russo_SwinUNet_S1_temporal_2025_2503.07230.pdf` | Modern **SAR-only** LC mapping (temporal S1) — for “improve MultiSenGE.” |

### Read next (same month)


| # | File | Why |
|---|------|-----|
| **5** | `papers/Prithvi-EO-2.0_2024_2412.02732.pdf` | Geospatial FM for **segmentation** — beat 2022 U-Net properly. |
| **6** | `papers/ChangeChat_2024_2409.08582.pdf` | Change **dialogue** pattern (mostly RGB bi-temp). |
| **7** | `papers/DeltaVLM_2025_2507.22346.pdf` | Stronger ChangeChat follow-up — read after #6. |

### Skim later (context only)


| # | File | Why |
|---|------|-----|
| **8** | `papers/EarthGPT_MMRS_2024_2401.16822.pdf` | Earlier multi-sensor VLM — background before EarthDial. |
| **9** | `papers/GeoChat_CVPR2024_2311.15826.pdf` | RGB grounding pioneer — history only, not SAR LULC. |

### If you only have time for 3

1. **EarthDial**  
2. **MultiSenGE** (`AI4LCC/multiSenge_AI4LCC.pdf`)  
3. **SARChat-2M** (SAR + language) *or* **Russo S1 temporal** (SAR + mapping / improve MultiSenGE)

Do not start with GeoChat or EarthGPT.

---

*Document: 2026-07-21 · Updated 2026-07-24 (reading order) · Independent research brief · MultiSenGE (CNN/mapping) and EarthDial (SAR VLM) treated separately.*