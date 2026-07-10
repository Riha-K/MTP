# Which Files to Read Now

> **Project:** AI4LCC-S1 VLM / LULCDial-S1  
> **Workspace:** `e:\MTP\earth2\`

Here’s a **reading order for right now** — shortest path to knowing what to do next.

---

## Read these first (today)

| # | File | Why |
|---|------|-----|
| **0** | [`RUNBOOK.md`](RUNBOOK.md) | **Main command file** — 1A data → **1B full-801 ZS** → **1C scale 25/50/100%** → eval → MultiSenNA |
| **1** | [`AI4LCC_S1_VLM_MTech_3Stage_Roadmap.md`](AI4LCC_S1_VLM_MTech_3Stage_Roadmap.md) | **Main plan** — 3 stages, novelty, metrics, what NOT to do, immediate P0 tasks |
| **2** | [`BenchmarkGuide/AI4LCC/BareSoil_AI4LCC_Workflow_Guide.md`](BenchmarkGuide/AI4LCC/BareSoil_AI4LCC_Workflow_Guide.md) | **How to run the pipeline** — download → parse JSON → VH tif → shards → train |
| **3** | [`LULCDial-s1/baresoil/README.md`](LULCDial-s1/baresoil/README.md) | **Data-prep commands** — shards, bench, zero-shot, MultiSenNA |

After these, you know: *what the thesis is*, *how data flows*, and *what to run*.

---

## Read when you need detail (this week)

| # | File | When |
|---|------|------|
| **4** | [`BenchmarkGuide/AI4LCC/MultiSenGE_AI4LCC_Complete_Analysis.md`](BenchmarkGuide/AI4LCC/MultiSenGE_AI4LCC_Complete_Analysis.md) | Before meeting supervisor — dataset facts, 14 classes, folder layout |
| **5** | [`BenchmarkGuide/AI4LCC/multiSenge_AI4LCC.pdf`](BenchmarkGuide/AI4LCC/multiSenge_AI4LCC.pdf) | Original paper — cite in thesis; skim abstract + § on data structure |
| **6** | Code (skim, don’t read line-by-line): `LULCDial-s1/baresoil/patch_meta.py`, `instruct_templates.py`, `build_instruct_s1.py` | When updating templates to **14 OCSGE class names** |

---

## Read later (not now)

| File | When |
|------|------|
| [`EarthDial_Complete_Analysis.md`](EarthDial_Complete_Analysis.md) | Before fine-tuning — EarthDial stages, tokens, dataloader |
| [`LULCDial-s1/README.md`](LULCDial-s1/README.md) | When setting up env / GPU training |
| [`LULCDial-s1/src/shell/data/Stage4_BareSoil_S1.json`](LULCDial-s1/src/shell/data/Stage4_BareSoil_S1.json) | Right before `finetune.py` |
| EarthDial eval READMEs (`rs_classification`, etc.) | When running baseline + eval |

---

## Skip for now

- Anything under `LULCDial-s1/src/vela_setup/` — IBM cluster templates, not your project
- Deleted benchmark guides (BigEarthNet, SARChat, etc.) — not on your path anymore

---

## One-line answer

**Start with:** `RUNBOOK.md` for commands → `AI4LCC_S1_VLM_MTech_3Stage_Roadmap.md` for plan.

**Your action after reading:** download **`s1.tgz`**, then follow **`RUNBOOK.md`** Step 1A.
