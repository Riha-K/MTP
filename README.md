# MTP / earth2

> **Story:** MultiSenGE **validation** (CNN: ConvLSTM+Inception-S1S2) + **VLM extension** (LULCDial-S1 / SAR-LC-Bench).  
> **Base paper:** MultiSenGE / AI4LCC — not EarthDial.  
> **Next:** replicate ConvLSTM+Inception → VGG-16 U-Net, then advanced model.

---

## Start here

| # | File |
|---|------|
| **1** | [`ROADMAP.md`](ROADMAP.md) — whole-project plan (Pillar A next) |
| **2** | [`log.md`](log.md) — history |
| **3** | [`multisenge_seg/`](multisenge_seg/) — CNN replicate + [`PROTOCOL.md`](multisenge_seg/PROTOCOL.md) |
| **4** | [`BenchmarkGuide/MultiSenGE_Validation_and_VLM_Plan.md`](BenchmarkGuide/MultiSenGE_Validation_and_VLM_Plan.md) — paper survey |
| **5** | [`LULCDial-s1/RUNBOOK.md`](LULCDial-s1/RUNBOOK.md) — VLM / PARAM commands (Pillar B) |
| **6** | [`sar_lc_bench_v0.1/`](sar_lc_bench_v0.1/) — public bench package draft |

---

## Pillar B main result (v0.1, frozen)

| Model | Example F1 |
|-------|------------|
| EarthDial ZS | 0.052 |
| LULCDial_S1_v0.1 | **0.812** |
| → MultiSenNA transfer | **0.679** |
