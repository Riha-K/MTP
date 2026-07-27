# LULCDial-S1 — One Project Roadmap

> **This is the only plan doc.** Everything else supports it.  
> **Commands:** [`RUNBOOK.md`](RUNBOOK.md) · **History:** [`log.md`](log.md)

---

## 1. Project objective (read this first)

**One project, one story:**

Build and publish **SAR-LC-Bench + LULCDial-S1** — the first **Sentinel-1 VH** vision-language benchmark on **official 14-class OCSGE** (AI4LCC MultiSenGE), with **multi-label classification + 2-turn dialogue**, and show that **EarthDial must be fine-tuned** for this task (zero-shot fails; fine-tune succeeds).

**You are NOT doing (unless supervisor changes plan later):**
- CNN / U-Net vs VLM comparison  
- Speckle adapters, pixel masks, S1 vs S2 cross-sensor, full-year temporal stacks  
- Those are other papers — not this thesis core  

**Success = three phases done:**

| Phase | Name | Difficulty | Status |
|-------|------|------------|--------|
| **Phase 1** | Finish LULCDial + write results | Easiest (~80% done) | 🔄 write-up left |
| **Phase 2** | Package **SAR-LC-Bench v0.2** (public eval) | Easy–medium | ⏳ not packaged |
| **Phase 3** | Bi-temporal SAR change QA (small add-on) | Medium (optional) | ⏳ not started |

Phases 1 → 2 → 3 are **one thesis project**, not three separate theses.

---

## 2. Papers to read — order + objective

Path: `BenchmarkGuide/papers/` (MultiSenGE: `BenchmarkGuide/AI4LCC/multiSenge_AI4LCC.pdf`).

### Must read (this week)

| # | Paper | **Objective — what you should understand after reading** |
|---|--------|----------------------------------------------------------|
| **1** | `EarthDial_CVPR2025_2412.15190.pdf` | How EarthDial trains (3 stages), what SAR it actually covers (ships/disaster, not OCSGE LULC), fusion limits, what you extend |
| **2** | `multiSenge_AI4LCC.pdf` | What MultiSenGE provides (S1+S2+labels, 14 OCSGE, 2020 stacks), what their 2022 U-Net baseline did **not** finish (multitemporal DL) |
| **3** | `SARChat-2M_2025_2502.08168.pdf` | How SAR+language benchmarks are built; why their tasks are **objects** not land-cover — positions your work |

**If only 3 papers:** read **1 → 2 → 3**.

### Read next (when writing Phase 1 paper)

| # | Paper | **Objective** |
|---|--------|---------------|
| **4** | `ChangeChat_2024_2409.08582.pdf` | Template for **change dialogue** — use only if you do Phase 3 |
| **5** | `DeltaVLM_2025_2507.22346.pdf` | Stronger change-VLM design — after ChangeChat |

### Skim only (background, not required for core)

| # | Paper | **Objective** |
|---|--------|---------------|
| **6** | `EarthGPT_MMRS_2024_2401.16822.pdf` | What existed before EarthDial |
| **7** | `GeoChat_CVPR2024_2311.15826.pdf` | RGB grounding — not your modality |

**Skip for now:** Russo S1-CNN, Prithvi — only if supervisor asks for a **separate** mapping/CNN paper.

---

## 3. What is already done (your project so far)

All of this belongs to **Phase 1** (implementation). You still need the **write-up**.

| Done | Detail | Proof |
|------|--------|--------|
| S1 VH pipeline + OCSGE 14-class templates | classify + 2-turn dialogue | `baresoil/` |
| Instruction shards + 70/30 split | v0.2 | `shards/`, `train_ratio=0.7` |
| Test bench | **2497** patches | `bench/v0.2/ai4lcc_test.jsonl` |
| EarthDial zero-shot | example F1 **0.019** | `metrics/v0.2/earthdial_zs_baseline.json` |
| LULCDial_S1_v0.2 fine-tune | example F1 **0.800**, T1/T2 **0.121 / 0.364** | job 89647, `metrics/v0.2/lulcdial_v0.2.json` |
| MultiSenNA transfer (extra) | F1 **0.670**, no NA training | `metrics/v0.1/lulcdial_v0.1_multisenna.json` |
| Stage 1 scaling (historical) | 801 val, ZS→0.799 | `metrics/v0.1/` — **appendix only** |

---

## 4. Contradictions / things to be honest about

| Issue | What happened | What to do in thesis |
|-------|---------------|----------------------|
| **v0.1 vs v0.2 split** | Main numbers: **v0.2** (70/30, 2497). Old scaling used **801** val (90/10). | **Primary table = v0.2 only.** Put 801 scaling in appendix. |
| **MultiSenNA used v0.1 model** | Transfer eval used `LULCDial_S1_v0.1`, not v0.2 | Label row **“v0.1 GE → NA”** OR re-run predict with **v0.2** checkpoint (optional P1-D7) |
| **Classify strong, dialogue weak** | F1 **0.80** but turn-1 set-match **0.12** | Report both; primary metric = **example F1**; dialogue = secondary |
| **Single S1 date per patch** | Not full 2020 stack in training | Honest in methods; Phase 3 only adds **2-date** lite if you do it |
| **Not a new dataset** | AI4LCC/MultiSenGE exists | Claim: **new SAR-LC instruction + eval protocol**, not new satellite data |

No contradiction if you label splits and metrics clearly in one master table (§7).

---

## 5. Phase 1 — Finish LULCDial + write-up (~80% done)

**Goal:** Submittable chapter/paper with frozen v0.2 numbers.

### Already done ✅

- P1-A: Data, shards, bench v0.2, S1 TIFF pack  
- P1-B: EarthDial ZS, LULCDial v0.2 FT, predict, eval  
- P1-C: MultiSenNA transfer (supplementary row)

### Your steps now ⏳

| Step | Action | Output |
|------|--------|--------|
| **1.1** | Fill **master results table** (§7) | Thesis Table 1 / paper Table 1 |
| **1.2** | Write **2-page bench spec**: tasks, 14 classes, 70/30 split, metrics | Methods §3 |
| **1.3** | Pick **10 patches** — S1 preview + GT + model answer | Figures |
| **1.4** | Write **3–5 failure cases** (dialogue wrong, classify ok) | Discussion |
| **1.5** | Draft **Intro + Method + Results** (4–6 pages) | First manuscript |
| **1.6** | *(Optional)* Re-run MultiSenNA with **v0.2** model | Cleaner transfer row |

**Phase 1 done when:** manuscript draft exists + table matches git metrics.

---

## 6. Phase 2 — Package SAR-LC-Bench v0.2

**Goal:** Others can download bench + run `eval_zero_shot.py` — **no new training**.

| Step | Action | Output |
|------|--------|--------|
| **2.1** | Create folder `data/baresoil_s1/sar_lc_bench_v0.2/` with bench JSONL + README | Public layout |
| **2.2** | Document eval: `python -m baresoil.eval_zero_shot ...` | `EVAL_PROTOCOL.md` |
| **2.3** | Add **leaderboard.csv** (columns: model, F1, T1, T2) | Empty template + your rows filled |
| **2.4** | Write `BENCH_MANIFEST_v0.2.json` (paths, split rule, n=2497) | Reproducibility |
| **2.5** | Git tag / Zenodo when supervisor approves | Release |

**Eval tracks (all in one bench):**

| Track | Task | Metric |
|-------|------|--------|
| E1 | Multi-label classify | **example F1** (primary) |
| E2 | Dialogue turn 1 | set-match accuracy |
| E3 | Dialogue turn 2 | set-match accuracy |
| E4 | GE → MultiSenNA transfer | same metrics (supplementary) |

**Phase 2 done when:** README + manifest + one-command eval documented.

---

## 7. Phase 3 — Bi-temporal SAR change QA (optional stretch)

**Only after Phase 1–2.** Small add-on subsection in paper — **not** full-year stacks.

| Step | Action |
|------|--------|
| **3.1** | Pick **2 S1 dates** per patch from JSON (`corresponding_s1`) |
| **3.2** | Build `ai4lcc_change_smoke.jsonl` (~100 patches) |
| **3.3** | Add QA: “Did backscatter change?” + simple GT note |
| **3.4** | Run ZS + optional short FT; **5 examples** in paper |

**Phase 3 done when:** one small table + honest limitation paragraph.

---

## 8. Master results table (Phase 1 step 1.1)

**Primary story = v0.2 (70/30, n=2497).**

| Model | Split | N | Example F1 | T1 | T2 |
|-------|-------|---|------------|----|----|
| EarthDial ZS | 70/30 test | 2497 | **0.019** | 0.000 | 0.000 |
| **LULCDial_S1_v0.2** | 70/30 test | 2497 | **0.800** | 0.121 | 0.364 |
| LULCDial_S1_v0.1 → MultiSenNA *(supp.)* | transfer | 12115 | **0.670** | 0.018 | 0.081 |

---

## 9. Timeline

| Week | Do |
|------|-----|
| **1** | Read papers §2 (1–3); Phase 1 steps 1.1–1.2 |
| **2** | Phase 1 steps 1.3–1.5 (figures + draft) |
| **3** | Phase 2 steps 2.1–2.4 (bench package) |
| **4** | Submit draft OR Phase 3 smoke if time |

---

## 10. Files in this repo

| File | Use |
|------|-----|
| **`ROADMAP.md`** | This plan |
| **`RUNBOOK.md`** | PARAM / sbatch commands |
| **`log.md`** | What you ran |
| `BenchmarkGuide/AI4LCC/MultiSenGE_AI4LCC_Complete_Analysis.md` | Dataset lookup |
| `BenchmarkGuide/papers/` | PDFs (§2) |

---

## 11. Elevator pitch

> **SAR-LC-Bench v0.2** evaluates Sentinel-1 VH land-cover **classification and dialogue** on official **14-class OCSGE** (2497 test patches). **LULCDial-S1** fine-tunes EarthDial and raises example F1 from **0.02 to 0.80**, with optional regional transfer to MultiSenNA.

---

*Updated 2026-07-27 — single project, three phases only.*
