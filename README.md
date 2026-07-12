# Which Files to Read Now

> **Project:** AI4LCC-S1 VLM / LULCDial-S1  
> **Workspace:** `e:\MTP\earth2\`  
> **Status (2026-07-12):** **Stage 1 scaling DONE** — ZS **0.019** · p25 **0.782** · p50 **0.783** · **100% 0.799** · **next = MultiSenNA**

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
| Eval metrics (scaling curve) | `metrics/v0.1/earthdial_zs_baseline.json`, `lulcdial_p25.json`, `lulcdial_p50.json`, `lulcdial_v0.1.json` |
| Train metrics | `metrics/v0.1/train_p25/`, `train_p50/`, `train_v0.1/` |

---

## Done vs next

| Stage | Status |
|-------|--------|
| 1A shards + GE/NA benches | **DONE** |
| 1B EarthDial ZS (801) | **DONE** — F1 ≈ **0.019** |
| 1C-a 25% + 1D | **DONE** — F1 **0.782** |
| 1C-b 50% + 1D | **DONE** — F1 **0.783** |
| 1C-c 100% + 1D | **DONE** — F1 **0.799** |
| MultiSenNA transfer | **NEXT** (Stage 2) |

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
- Template/scorer redesign — optional; scaling curve is done

---

## One-line answer

**Commands → `RUNBOOK.md`.** **History → `log.md`.** **Next → MultiSenNA transfer eval with best GE model (`LULCDial_S1_v0.1`).**
