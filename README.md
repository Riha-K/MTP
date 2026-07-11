# Which Files to Read Now



> **Project:** AI4LCC-S1 VLM / LULCDial-S1  

> **Workspace:** `e:\MTP\earth2\`  

> **Status (2026-07-11):** **1A + 1B done** · **next = 1C-a (25% fine-tune on PARAM)**



---



## Read these first (today)



| # | File | Why |

|---|------|-----|

| **0** | [`RUNBOOK.md`](RUNBOOK.md) | **Main command file** — status checklist, **PARAM env pins**, 1C subsample + FT |

| **1** | [`log.md`](log.md) | **What was done** (newest first) — ZS numbers, PARAM installs |

| **2** | [`LULCDial-s1/baresoil/README.md`](LULCDial-s1/baresoil/README.md) | Short data-prep / eval commands |

| **3** | [`Stage1_Summer_Intern_Guide.md`](Stage1_Summer_Intern_Guide.md) | Stage goals + glossary |



After these, you know: *where we are*, *what to run next*, *PARAM setup*.



---



## Where everything is written



| Need | File |

|------|------|

| Copy-paste commands + PARAM GPU env | **`RUNBOOK.md`** |

| History / results / “what we did yesterday” | **`log.md`** |

| Thesis plan (3 stages) | **`AI4LCC_S1_VLM_MTech_3Stage_Roadmap.md`** |

| Fine-tune shard paths | **`LULCDial-s1/src/shell/data/Stage4_BareSoil_S1.json`** |

| ZS baseline metrics | **`LULCDial-s1/data/baresoil_s1/metrics/v0.1/earthdial_zs_baseline.json`** |

| ZS predictions | **`LULCDial-s1/data/baresoil_s1/bench/v0.1/preds/earthdial_zs/`** |



---



## Done vs next



| Stage | Status |

|-------|--------|

| 1A shards + GE/NA benches | **DONE** (on PARAM) |

| 1B EarthDial ZS (801) | **DONE** — F1 ≈ **0.0194** |

| 1C-a 25% fine-tune | **NEXT** — subsample uploaded train shard, then FT |

| 1C-b / 1C-c | after 1C-a |



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



**Commands + PARAM env → `RUNBOOK.md`.** **History → `log.md`.** **Next action → subsample train to p25 on PARAM, then fine-tune.**


