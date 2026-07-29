# LULCDial-S1 / AI4LCC-S1 VLM

> **One project:** SAR-LC-Bench + LULCDial-S1 (Sentinel-1 VH · OCSGE 14-class · EarthDial)  
> **Status:** v0.1 metrics done · MultiSenNA transfer done · **Next:** write-up

---

## Start here

| # | File |
|---|------|
| **1** | [`ROADMAP.md`](ROADMAP.md) — objective, papers, phases |
| **2** | [`RUNBOOK.md`](RUNBOOK.md) — PARAM commands |
| **3** | [`log.md`](log.md) — history |

---

## Main result (v0.1, 2497 test, 70/30)

| Model | Example F1 |
|-------|------------|
| EarthDial ZS | 0.019 |
| LULCDial_S1_v0.1 | **0.800** |
| LULCDial_S1_v0.1 → MultiSenNA (transfer, no retrain) | **0.674** |

Metrics: `LULCDial-s1/data/lulcdial_s1/metrics/v0.1/`
