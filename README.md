# LULCDial-S1 / AI4LCC-S1 VLM

> **One project:** SAR-LC-Bench + LULCDial-S1 (Sentinel-1 VH · OCSGE 14-class · EarthDial)  
> **Status:** Phase 1 complete · Dialogue soft F1 ✅ · Format-aligned dialogue FT optional next · Zenodo deferred

---

## Start here

| # | File |
|---|------|
| **1** | [`ROADMAP.md`](ROADMAP.md) — objective, papers, phases |
| **2** | [`RUNBOOK.md`](RUNBOOK.md) — PARAM commands |
| **3** | [`log.md`](log.md) — history |
| **4** | [`LULCDial-s1/docs/DIALOGUE_IMPROVE.md`](LULCDial-s1/docs/DIALOGUE_IMPROVE.md) — dialogue track |
| **5** | [`sar_lc_bench_v0.1/`](sar_lc_bench_v0.1/) — public package draft |

---

## Main result (v0.1, 2497 test, 70/30, post-radiometry-fix)

| Model | Example F1 |
|-------|------------|
| EarthDial ZS | 0.052 |
| LULCDial_S1_v0.1 | **0.812** |
| LULCDial_S1_v0.1 → MultiSenNA (transfer, no retrain) | **0.679** |

Metrics: `LULCDial-s1/data/lulcdial_s1/metrics/v0.1/`
