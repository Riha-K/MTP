# CNN validation protocol — ConvLSTM+Inception-S1S2 replication

> **Paper:** Wenger et al., Remote Sensing 2023 (`BenchmarkGuide/Wenger_etal_2023_ConvLSTM_Inception_MultiSenGE_RemoteSensing.pdf`)  
> **Goal:** Match their setting as closely as possible; then swap in an advanced model.  
> **Code:** `multisenge_seg/` (this package). **Data:** `LULCDial-s1/data/lulcdial_s1/ai4lcc/multisenge/`

---

## 1. Data roots (local)

```text
…/ai4lcc/multisenge/
  labels/             # 8157 JSON
  s1/                 # ~110 GB
  s2/                 # ~88 GB  ✅ downloaded
  ground_reference/   # 8157 GR TIFFs ✅ downloaded
```

---

## 2. Temporal selection (paper)

| Item | Value |
|------|--------|
| Months | **July, August, September, November** (2020) |
| Constraint | ≥ **17 days** between consecutive selected S2 dates across months |
| Depth | **4** dates (one per month when available) |
| S1 | Matching multitemporal stack (paper uses S1 branches aligned to the series) |

Exact patch filtering follows paper months + ≥17-day S2 gap among **on-disk** dates;
S1 is paired per S2 date (same month if available, else nearest within 45 days).
Scaffold implements this; full index: `python -m multisenge_seg.smoke_index` (slow once);
quick: `python -m multisenge_seg.smoke_index --max-labels 300`.

---

## 3. Geographic split (paper Fig. 4)

| Split | Sentinel-2 tiles |
|-------|------------------|
| **Train** | T32UMV, T32ULU, T32ULV, T32TLT, T31UGQ, T31TFN, T31UFQ, T31UFR |
| **Val** | T31UFP, T31UGP |
| **Test** | T31UEQ |

Tile string is the prefix of patch ids (e.g. `31TFN_4626_514` → tile `31TFN`).  
Paper writes `T31TFN`-style names; our filenames omit the leading `T` (`31TFN_…`).  
Note: paper body omits **T32ULV** in the prose list; including it matches published **n_train=3369**.

---

## 4. Class taxonomies

**Native labels:** OCSGE IDs 1–14 in `ground_reference` rasters.

### 6-class (urban focus)

| Out id | Meaning |
|--------|---------|
| 1–5 | Urban fabrics unchanged |
| 6 | All other (ids 6–14 merged) |

### 10-class (paper merge)

| Out id | Source IDs |
|--------|------------|
| 1–6 | same as native 1–6 |
| 7 | Vineyards+Orchards (7∪8) |
| 8 | Grasslands (9) |
| 9 | Forests + hedges/hedges + mineral (10∪11∪12) |
| 10 | Wetlands + Water (13∪14) |

Ignore / mask nodata as documented in loaders (0 or blank).

**Default first run:** **6-class** (simpler metrics vs urban story). Flag `--num-classes 10` for second run.

---

## 5. Model (replicate)

```text
S1 time series ──► ConvLSTM ──┐
S2 time series ──► ConvLSTM ──┼── concat ──► U-Net (VGG-16 encoder) ──► HxW logits
First-date S1+S2 ► Inception ─┘
```

Paper notes: ConvLSTM kernel **3×3**, **32** filters; naive Inception **1×1 / 3×3 / 5×5** + pool;
U-Net with **VGG-16** encoder; weighted categorical CE; Adam LR **1e−3**; ReduceLROnPlateau (×0.1, patience 5);
EarlyStopping patience **20**; geometric aug (~75%); multitemporal channel mean/std norm.

Weights were **not** released publicly — reimplement or obtain on demand.

---

## 6. Metrics

Computed in `metrics.py` (`scores_from_cm`) from pixel confusion matrix:

- Per-class **Precision / Recall / F1** (paper Table 5)
- Support-weighted **W-Precision / W-Recall / W-F1** (paper “W-Avg” row)
- **Cohen’s Kappa** (paper Table 6)
- Accuracy (logged; not paper headline)

Frozen replicate vs paper: [`RESULTS_RS2023_6CLASS.md`](RESULTS_RS2023_6CLASS.md).

**Never** compare these numbers to LULCDial **patch example F1**.

---

## 7. Machines

| Step | Where |
|------|--------|
| Protocol / code / smoke `--max-patches` | Laptop |
| Full tensor cache build (optional) | Sir PC or laptop if disk OK |
| Full train | **PARAM** |

---

## 8. Status

| Item | Status |
|------|--------|
| S1 / S2 / GR / labels on disk | ✅ |
| This protocol | ✅ drafted |
| Loader + VGG-16 U-Net model | ✅ `multisenge_seg/` |
| Train replicate (6-class) | ✅ report **v0 last.pt epoch 25** (test W-F1 0.9037) |
| 10-class replicate | ⬜ next (`--num-classes 10`) |
| Advanced model (U-TAE / etc.) | ⬜ after 10-class |
