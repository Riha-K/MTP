# Fair Comparison Plan — CNN vs LULCDial-S1

---

## 1. Target table 


| Method               | Task                             | Sensor               | Pixel wF1 | Patch example F1           | Dialogue T1 / T2 |
| -------------------- | -------------------------------- | -------------------- | --------- | -------------------------- | ---------------- |
| U-Net (Wenger-style) | pixel seg → optional tags        | S2 IRRG and/or S1 VH | **?**     | **?** (from map)           | —                |
| ResNet multi-label   | patch classify                   | S1 VH                | —         | **?**                      | —                |
| EarthDial ZS         | patch classify + dialogue (text) | S1 VH                | —         | **0.0187** (v0.2 / 2497)   | **0 / 0**        |
| **LULCDial-S1**      | patch classify + dialogue (text) | S1 VH                | —         | **?** (was 0.799 on 90/10) | **?**            |


---

## 3. Shared rules 

Everyone uses the **same MultiSenGE geography and OCSGE labels**, but not every model does every output.

### 3.1 Data


| Item      | Choice                                                          |
| --------- | --------------------------------------------------------------- |
| Region    | MultiSenGE (Grand-Est)                                          |
| Labels    | Official OCSGE **14** classes (IDs 1–14)                        |
| Split     | **70% train / 30% test** (MD5 of patch stem, `train_ratio=0.7`) |
| Test set  | `bench/v0.2/ai4lcc_test.jsonl` (~**2497** patches)              |
| Train set | Same MD5 train bucket as shards `ai4lcc_ge_train_train`         |




### 3.2 Two metrics (two worlds)


| Metric                | What it measures                                          | Who reports it                                                           |
| --------------------- | --------------------------------------------------------- | ------------------------------------------------------------------------ |
| **Pixel weighted F1** | Every pixel’s class vs ground-reference map               | **U-Net only** (native output)                                           |
| **Patch example F1**  | Predicted **set of class names** vs GT set for that patch | **ResNet, EarthDial, LULCDial**, and U-Net **after aggregating** the map |


**Patch example F1** = same formula as `eval_zero_shot.py` (mean of per-patch set F1 on class names).

### 3.3 How U-Net gets a patch example F1

```text
U-Net predicts H×W class map
        ↓
for each class c: if fraction of pixels with class c > τ  →  include class c
        ↓
class name list  →  same JSONL-style scoring as LULCDial
```

Suggested threshold: **τ = 0.01 or 0.05** (report which one; sensitivity check optional).

---



## 4. Method A — ResNet multi-label (do first)

**Goal:** Same *classify* exam as LULCDial, with a CNN instead of a VLM.


| Item      | Spec                                                               |
| --------- | ------------------------------------------------------------------ |
| Input     | S1 VH 256×256 (same dB clip as LULCDial: ≈ **[-50, +10]**)         |
| Model     | ResNet-18 or ResNet-50 (ImageNet init), head = **14 × sigmoid**    |
| Loss      | BCEWithLogits                                                      |
| Train     | MD5 **70%** train patches only                                     |
| Predict   | Prob ≥ 0.5 → class present → OCSGE names                           |
| Score     | Write `pred_classify` JSONL → `eval_zero_shot.py` → **example F1** |
| Pixel wF1 | — (no map)                                                         |
| Dialogue  | —                                                                  |


**Why this matters for the professor:**  
“On the **same patch multi-label task**, here is CNN vs VLM.”

**Effort:** ~1 week.

---



## 5. Method B — U-Net Wenger-style (do second)

**Goal:** Replicate *their kind of model* on *our labels/split*, then bridge to our patch metric.

### 5.1 Train


| Item                              | Spec                                                                                |
| --------------------------------- | ----------------------------------------------------------------------------------- |
| Target                            | `ground_reference` GeoTIFF (pixel class IDs)                                        |
| Input options (pick at least one) | **B1:** S2 IRRG (closest to paper U-Net-IRRG) · **B2:** S1 VH (fair to your thesis) |
| Architecture                      | U-Net (VGG/ResNet encoder OK)                                                       |
| Loss                              | Weighted cross-entropy (class imbalance)                                            |
| Split                             | Same **70/30** patch list as LULCDial                                               |




### 5.2 Report


| Output                    | Metric                              |
| ------------------------- | ----------------------------------- |
| Pixel map on test patches | **Weighted pixel F1** (their world) |
| Map → class set with τ    | **Patch example F1** (your world)   |


**Why this matters:**  
You show you *can* run their technique with your labels — without claiming EarthDial is a U-Net.

**Effort:** ~2 weeks after ResNet.

---



## 6. Method C — EarthDial ZS + LULCDial (already in progress)


| Step                                                 | Status                                      |
| ---------------------------------------------------- | ------------------------------------------- |
| Rebuild 70/30 shards + bench v0.2                    | ✅ Done                                      |
| Pack `s1_test_bench_v0.2` + upload PARAM             | ✅ Done                                      |
| EarthDial ZS on 2497                                 | ✅ Done — example F1 **0.0187**, dial **0/0** |
| Score ZS → `metrics/v0.2/earthdial_zs_baseline.json` | ✅ Done                                      |
| Fine-tune `LULCDial_S1_v0.2`                         | 🔄 Job **89631** (`train_v0.2.sbatch`)       |
| Predict + score LULCDial on same 2497                | ⏳ After FT                                  |


Fill table cells for **example F1** + dialogue set-match from these runs.

---



## 7. What we will **not** claim

1. **Do not** say LULCDial “beats” U-Net because example F1 > paper’s Metz pixel F1.
2. **Do not** force Phi-3 to output a dense pixel map (not Stage 1 scope).
3. **Do not** mix **v0.1 (801)** and **v0.2 (2497)** numbers in one leaderboard without clear labels.
4. **Do not** put dialogue scores on ResNet/U-Net rows (unless you later bolt on a language head — out of scope).

---



## 8. What we **will** claim (if numbers cooperate)


| Claim                                      | Evidence                                                                         |
| ------------------------------------------ | -------------------------------------------------------------------------------- |
| VLM needs adaptation for S1 OCSGE dialogue | EarthDial ZS example F1 ≈ floor → LULCDial much higher on **same** bench         |
| Fair classify comparison                   | ResNet vs LULCDial **example F1** on same 2497                                   |
| Mapping vs interaction                     | U-Net strong on **pixel wF1**; LULCDial strong on **text + dialogue + transfer** |
| Protocol contribution                      | AI4LCC → instruction/dialogue bench (complementary to CNN papers)                |


Honest outcomes are fine: e.g. ResNet > LULCDial on tags, or U-Net >> both on pixels — still publishable if framed clearly.

---



## 9. Work order (recommended)

```text
[Now]     Fine-tune LULCDial_S1_v0.2 (job 89631) → then pred + score
[Done]    v0.2 EarthDial ZS example F1 ≈ **0.0187** (same floor as 90/10 ≈ 0.019)
[Week+1]  ResNet S1 multi-label on same 70/30 → example F1
[Week+2]  U-Net on ground_reference (S1 and/or S2)
            → pixel wF1 + aggregated example F1
[Then]    Fill one table + short “apples vs oranges” paragraph for thesis/slides
```

Optional later: MultiSenNA row for LULCDial only (no NA train) — CNN baseline train-GE-only transfer can wait.

---



## 10. Agree / disagree checklist

Mark each item when you read this:


| #   | Decision                                                                                 | Agree? (Y/N) |
| --- | ---------------------------------------------------------------------------------------- | ------------ |
| 1   | Primary shared metric for VLM↔CNN classify = **patch example F1** on `ai4lcc_test.jsonl` |              |
| 2   | U-Net also reports **pixel weighted F1** in a separate column                            |              |
| 3   | First CNN baseline = **ResNet multi-label on S1** (not full Metz replication)            |              |
| 4   | Second = **U-Net** on ground_reference with same 70/30                                   |              |
| 5   | Prefer U-Net input first: **S1 VH** / **S2 IRRG** / **both** (circle one)                |              |
| 6   | Aggregation threshold τ = **0.01** / **0.05** (circle one)                               |              |
| 7   | Keep dialogue columns for VLM only                                                       |              |
| 8   | All new numbers on **70/30 v0.2**; archive 90/10 as historical                           |              |


---



## 11. One paragraph for your professor

> We will not compare LULCDial example F1 to the MultiSenGE paper’s Metz U-Net pixel F1 as a single leaderboard. We will (1) run EarthDial ZS and LULCDial on our **70/30** MultiSenGE S1 dialogue bench with **example F1**; (2) train a **ResNet multi-label** baseline on the same patches and metric; (3) train a **U-Net** on the same OCSGE ground-reference maps, report **pixel weighted F1**, and convert maps to patch class sets for **example F1**. That gives one table covering mapping and interactive classify/dialogue without mixing incompatible scores.

---



## 12. Open questions for you

1. Is **ResNet-first** OK, or does he insist **U-Net first**?
2. For U-Net, must we use **S2** to match the PDF, or is **S1-only** enough for the thesis story?
3. Do you want MultiSenNA in this table, or only MultiSenGE for the CNN vs VLM block?

---

*After you mark §10, we implement Path A (ResNet) or Path B (U-Net) next — VLM v0.2 continues in parallel on PARAM.*