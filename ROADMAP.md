# LULCDial-S1 — One Project Roadmap

> **This is the only plan doc.** Everything else supports it.  
> **Commands:** `[RUNBOOK.md](RUNBOOK.md)` · **History:** `[log.md](log.md)`

---

## 1. Project objective

**One project, one story:**

Build and publish **SAR-LC-Bench + LULCDial-S1** — Sentinel-1 VH vision-language benchmark on **official 14-class OCSGE** (AI4LCC MultiSenGE), with **multi-label classification + 2-turn dialogue**, and show that **EarthDial must be fine-tuned** (zero-shot fails; fine-tune succeeds).

**Publishable claim (not weak if you finish the package):**


| What you have                                                           | Why it matters                                               |
| ----------------------------------------------------------------------- | ------------------------------------------------------------ |
| **First** SAR-VLM bench on **OCSGE 14-class** + **classify + dialogue** | SARChat/SAREval = objects/ships; not this taxonomy           |
| **2497** held-out test, **70/30** split, frozen eval protocol           | Reproducible, not a one-off demo                             |
| **ZS 0.02 → FT 0.80** on same bench                                     | Clear adaptation story                                       |
| **GE → MultiSenNA transfer** (re-run with v0.1 checkpoint)              | Regional generalization                                      |
| **Public bench release** (Phase 2)                                      | What makes it a *benchmark paper*, not just a fine-tune note |


**To strengthen before submission:** package bench + eval script, 10 figures, failure analysis, **one** clean results table (no mixed splits), optional bi-temporal subsection.

---



## 2. Versioning 


| Item               | Path / name                                           |
| ------------------ | ----------------------------------------------------- |
| Bench              | `bench/v0.1/ai4lcc_test.jsonl` (**2497** test, 70/30) |
| S1 TIFF pack       | `ai4lcc/multisenge/s1_test_bench_v0.1/`               |
| Metrics            | `metrics/v0.1/`                                       |
| Checkpoint (PARAM) | `checkpoints/LULCDial_S1_v0.1`                        |
| Model tag          | `LULCDial_S1_v0.1`                                    |


**No v0.2 label anywhere.** Old 90/10 (801 val) and scaling runs are **removed** from repo.

---



## 3. Papers to read 

### Must read (this week)


| #     | Paper          | Objective                                             |
| ----- | -------------- | ----------------------------------------------------- |
| **1** | EarthDial      | Training stages, SAR coverage gap, what you extend    |
| **2** | MultiSenGE PDF | Dataset, 14 OCSGE, what U-Net baseline did not finish |
| **3** | SARChat-2M     | How SAR+language benches are built; object vs LULC    |


### When writing


| #     | Paper      | Objective                               |
| ----- | ---------- | --------------------------------------- |
| **4** | ChangeChat | Change dialogue template (Phase 3 only) |
| **5** | DeltaVLM   | Stronger change-VLM design              |


---

## 4. Done (v0.1)


| Done                             | Detail                                       | Proof                                     |
| -------------------------------- | -------------------------------------------- | ----------------------------------------- |
| S1 VH + OCSGE 14-class templates | classify + 2-turn dialogue                   | `lulcdial/`                               |
| 70/30 shards + bench             | train_ratio=0.7                              | `shards/`, `bench/v0.1/`                  |
| Test bench                       | **2497** patches                             | `bench/v0.1/ai4lcc_test.jsonl`            |
| EarthDial ZS                     | example F1 **0.052**                         | `metrics/v0.1/earthdial_zs_baseline.json` |
| LULCDial v0.1 FT                 | F1 **0.812**; T1/T2 set **0.134 / 0.390**; T1/T2 F1 **0.813 / 0.870** | `metrics/v0.1/lulcdial_v0.1.json` |
| MultiSenNA transfer              | F1 **0.679**; T1/T2 set **0.013 / 0.079**; T1/T2 F1 **0.687 / 0.686** | `metrics/v0.1/lulcdial_v0.1_multisenna.json` |


---

## 5. Honest limitations (report in paper)


| Issue                          | What to do                              |
| ------------------------------ | --------------------------------------- |
| Classify strong, dialogue **set-match** weak | Report **both** set-match and dialogue **example F1**; primary classify = example F1. Soft T1 F1 ≈ 0.81 on GE while set-match ≈ 0.13 (format/consistency gap). Plan: `LULCDial-s1/docs/DIALOGUE_IMPROVE.md` |
| Single S1 date per patch       | Methods §; Phase 3 = 2-date lite only   |
| Not a new satellite dataset    | Claim = **instruction + eval protocol** |
| **Mixed VH radiometry (FIXED)** | Unconditional linear→dB in `s1_vh_io.py`. Shards rebuilt + model re-trained. Post-fix: ZS **0.052** / FT **0.812** / MultiSenNA **0.679** (was 0.019 / 0.800 / 0.674). |


---

## 6. Phase 1 — Write-up


| Step    | Action                                                | Status |
| ------- | ----------------------------------------------------- | ------ |
| **1.1** | Master results table (§10)                            | ✅ local `writeup/` |
| **1.2** | Bench spec (tasks, 14 classes, 70/30, metrics)        | ✅ local `writeup/` |
| **1.3** | 10 figure patches                                     | ✅ 10/10 PNGs in local `writeup/figures/v0.1/` |
| **1.4** | 3–5 failure cases                                     | ✅ 5 cases, real ZS/FT preds |
| **1.5** | Draft Intro + Method + Results                        | ✅ local `writeup/` (+ radiometry caveat) |
| **1.6** | **MultiSenNA** predict + eval with `LULCDial_S1_v0.1` | ✅ F1 **0.679** transfer (11939 patches, post-fix) |

`writeup/` is **local only** (gitignored) — paper drafts, figures, Table 1 template.


---

## 7. Phase 2 - Package SAR-LC-Bench v0.1

Public folder [`sar_lc_bench_v0.1/`](sar_lc_bench_v0.1/) drafted in this research repo:

| Artifact | Status |
|----------|--------|
| `README.md`, `EVAL_PROTOCOL.md`, `LICENSE` | ✅ |
| `BENCH_MANIFEST_v0.1.json` + SHA256 of GE test JSONL | ✅ |
| `leaderboard.csv` (post-fix numbers) | ✅ |
| `data/ge/ai4lcc_test.jsonl` (2497 rows) | ✅ |
| `PUBLISH.md` (separate public GitHub repo steps) | ✅ |
| Compact `s1_test_bench/` TIFF pack + Zenodo | ⏸ **deferred** — publish after Phase 3 (more data in one release) |
| Standalone public GitHub repo `SAR-LC-Bench` | ⏸ **deferred** — same |

**Decision (2026-08-03):** keep protocol draft here; go public (GitHub + free Zenodo TIFF pack) **at the end**, after bi-temporal Phase 3, so one release can include single-date + 2-date materials.

**Do not** open the whole MTP repo as the public bench — it has personal notes. See `sar_lc_bench_v0.1/PUBLISH.md`.

---

## 8. Phase 3 — Bi-temporal change QA (lite) — **NEXT**

**Goal:** small extension subsection — **2 Sentinel-1 dates**, ~**100** patches, classify/change dialogue, one small results table. Optional for core thesis; strengthens the temporal story.

### Scope (keep lite)

| Item | Choice |
|------|--------|
| Dates per patch | **Exactly 2** VH acquisitions (not full-year stack) |
| Patch count | ~**100** from MultiSenGE test (or train∩multi-date) |
| Labels | Same OCSGE multi-label presence; plus simple **change** QA (what appeared / disappeared / unchanged) |
| Model | Start with **ZS + existing LULCDial_S1_v0.1** (single-image FT may transfer poorly to 2-image input — report honestly); optional small FT later |
| Publish | Bundle 2-date pack with Phase 2 Zenodo **at end** |

### Machine

| Step | Where |
|------|--------|
| Discover patches with ≥2 S1 dates; write 2-date JSONL | **Sir PC** (full `multisenge/s1` needed) — laptop usually lacks full archive |
| Template + eval scripts | **Laptop** |
| GPU predict / optional FT | **PARAM** |

### First concrete steps

1. On sir PC: list patches that have ≥2 `*_S1_*.tif` dates; sample ~100.
2. Define instruction templates (ChangeChat-style, S1 VH).
3. Build `bench/v0.1/ai4lcc_bitemp_100.jsonl`.
4. Smoke predict 20 → full 100 → small table in writeup.

Papers (when writing): ChangeChat, DeltaVLM (templates only — RGB bi-temp; we adapt to S1).

---

## 9. After multitemporal — what next (if you want more)


| Add-on                                            | Required?               | Effort       | Publication value           |
| ------------------------------------------------- | ----------------------- | ------------ | --------------------------- |
| **MultiSenNA transfer** (v0.1 model)              | ✅ done                   | —            | Regional generalization row |
| **Bench package draft**                           | ✅ drafted                | —            | Protocol ready              |
| **Dialogue metric / format-aligned FT**       | **IN PROGRESS** (soft F1 ✅) | Soft = laptop; FT = sir+PARAM | Fixes weak set-match story |
| **Bi-temporal change QA** (2-date lite)       | After dialogue track        | ~1 week                      | Small extension subsection  |
| **Public GitHub + Zenodo TIFF packs**         | **Yes — at end**            | 1–2 days                     | Makes project publishable   |


**Order now:** **Dialogue soft metrics ✅ → optional dialogue FT → Phase 3 bi-temporal (optional) → public release**.

---


## 10. Master results table


| Model                         | Split      | N     | Example F1 | T1 set | T2 set | T1 F1 | T2 F1 |
| ----------------------------- | ---------- | ----- | ---------- | ------ | ------ | ----- | ----- |
| EarthDial ZS                  | 70/30 test | 2497  | **0.052**  | 0.000  | 0.000  | 0.000 | 0.000 |
| **LULCDial_S1_v0.1**          | 70/30 test | 2497  | **0.812**  | 0.134  | 0.390  | **0.813** | **0.870** |
| LULCDial_S1_v0.1 → MultiSenNA | transfer   | 11939 | **0.679**  | 0.013  | 0.079  | **0.687** | **0.686** |


Dialogue set-match is strict; dialogue example F1 is soft (same as classify F1 on that turn). See `LULCDial-s1/docs/DIALOGUE_IMPROVE.md`.

---



## 11. Elevator pitch

> **SAR-LC-Bench v0.1** evaluates Sentinel-1 VH land-cover **classification and dialogue** on official **14-class OCSGE** (2497 test patches). **LULCDial-S1** fine-tunes EarthDial and raises example F1 from **0.05 to 0.81** (post-radiometry-fix), and transfers to a different French region (MultiSenNA, Nouvelle-Aquitaine) with **no retraining** at F1 **0.68**.

---

*Updated 2026-07-27 — v0.1 only (70/30).*