# Which Files to Read Now

> **Project:** AI4LCC-S1 VLM / LULCDial-S1  
> **Workspace:** `e:\MTP\earth2\`  
> **Status (2026-07-11):** **1C-a + 1D p25 DONE** (F1 **0.782**) · **1C-b RUNNING overnight** (job **88490**)

---

## Read these first (today)

| # | File | Why |
|---|------|-----|
| **0** | [`RUNBOOK.md`](RUNBOOK.md) | **Main command file** — status checklist, PARAM env, `sbatch` FT / predict |
| **1** | [`log.md`](log.md) | **What was done** (newest first) — metrics, PARAM jobs |
| **2** | [`LULCDial-s1/baresoil/README.md`](LULCDial-s1/baresoil/README.md) | Short data-prep / eval commands |
| **3** | [`Stage1_Summer_Intern_Guide.md`](Stage1_Summer_Intern_Guide.md) | Stage goals + glossary |

After these, you know: *where we are*, *what to run next*, *PARAM setup*.

---

## Where everything is written

| Need | File |
|------|------|
| Copy-paste commands + PARAM GPU env | [`RUNBOOK.md`](RUNBOOK.md) |
| History / results | [`log.md`](log.md) |
| Thesis plan (3 stages) | [`AI4LCC_S1_VLM_MTech_3Stage_Roadmap.md`](AI4LCC_S1_VLM_MTech_3Stage_Roadmap.md) |
| Fine-tune meta (p25) | `LULCDial-s1/src/shell/data/Stage4_BareSoil_S1.json` |
| Fine-tune meta (p50) | `LULCDial-s1/src/shell/data/Stage4_BareSoil_S1_p50.json` |
| 1C-b train / predict launchers | `LULCDial-s1/src/shell/train_p50.sbatch`, `pred_p50.sbatch` |
| ZS baseline | `LULCDial-s1/data/baresoil_s1/metrics/v0.1/earthdial_zs_baseline.json` |
| p25 eval (vs ZS) | `LULCDial-s1/data/baresoil_s1/metrics/v0.1/lulcdial_p25.json` |
| p25 train metrics | `LULCDial-s1/data/baresoil_s1/metrics/v0.1/train_p25/` |

---

## Done vs next

| Stage | Status |
|-------|--------|
| 1A shards + GE/NA benches | **DONE** (on PARAM) |
| 1B EarthDial ZS (801) | **DONE** — F1 ≈ **0.0194** |
| 1C-a 25% fine-tune | **DONE** — `LULCDial_S1_p25/` |
| 1D p25 vs ZS | **DONE** — F1 **0.782** |
| 1C-b 50% fine-tune | **RUNNING overnight** — job **88490** → `LULCDial_S1_p50/` |
| 1D p50 eval | after 1C-b finishes → `pred_p50.sbatch` + `lulcdial_p50.json` |
| 1C-c 100% | after p50 |

---

## Read when you need detail

| # | File | When |
|---|------|------|
| **4** | [`BenchmarkGuide/AI4LCC/BareSoil_AI4LCC_Workflow_Guide.md`](BenchmarkGuide/AI4LCC/BareSoil_AI4LCC_Workflow_Guide.md) | PI-level data workflow |
| **5** | [`BenchmarkGuide/AI4LCC/MultiSenGE_AI4LCC_Complete_Analysis.md`](BenchmarkGuide/AI4LCC/MultiSenGE_AI4LCC_Complete_Analysis.md) | Dataset facts / 14 classes |
| **6** | `LULCDial-s1/baresoil/*.py` | When changing templates / builders |

---

## Skip for now

- `LULCDial-s1/src/vela_setup/` — IBM cluster templates
- Template/scorer redesign — wait until after 25/50/100% scaling curve

---

## One-line answer

**Commands + PARAM env → `RUNBOOK.md`.** **History → `log.md`.** **Morning → check job 88490 / `LULCDial_S1_p50`, then `pred_p50.sbatch` + score.**
