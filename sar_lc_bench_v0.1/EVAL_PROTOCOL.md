# Evaluation protocol — SAR-LC-Bench v0.1

## 1. Tasks

Each MultiSenGE test patch defines **three** prompts:

### A — Multi-label classification

- **Input:** Sentinel-1 VH GeoTIFF (256×256, 10 m) + classify instruction with the full class option list.
- **Output:** comma-separated OCSGE class names (all that apply).
- **Metric:** **example F1** — for each patch, F1 between predicted and GT class-name sets; mean over patches.

### B — Dialogue turn 1

- **Question:** list all land-cover classes present.
- **Metric:** **set-match accuracy** (exact set equality after normalization).

### C — Dialogue turn 2

- **Question:** which of these are natural or agricultural (class IDs 6–15).
- **Metric:** **set-match accuracy**.

Primary leaderboard column: **example F1**.

---

## 2. Taxonomy (14-class OCSGE, MultiSenGE)

| ID | Class name | Group |
|----|------------|-------|
| 1 | Dense Built-Up | Urban |
| 2 | Sparse Built-Up | Urban |
| 3 | Specialized Built-Up Areas | Urban |
| 4 | Specialized but Vegetative Areas | Urban |
| 5 | Large Scale Networks | Urban |
| 6 | Arable Lands | Agricultural |
| 7 | Vineyards | Agricultural |
| 8 | Orchards | Agricultural |
| 9 | Grasslands | Semi-natural |
| 10 | Groves and Hedges | Semi-natural |
| 11 | Forests | Semi-natural |
| 12 | Open Spaces, Mineral | Semi-natural |
| 13 | Wetlands | Wetlands |
| 14 | Water Surfaces | Water |

**MultiSenNA transfer** adds class **15 — Beaches, Sand** in the option list (not present in GE train labels).

---

## 3. Train / test split (frozen)

Deterministic 70/30 over MultiSenGE patch stems:

```text
h = int(md5(patch_stem), 16) % 1000
train if h < 700 else test
```

| Split | Patches | Role |
|-------|---------|------|
| Train | 5660 | Fine-tuning only (not scored as the official test) |
| Test | **2497** | Official in-domain leaderboard |

Test patches must not appear in training.

---

## 4. Radiometry (required for fair comparison)

AI4LCC S1 patches store **linear** VH intensity. Convert **unconditionally** to dB before the model:

```text
vh_db = 10 * log10(clip(vh_linear, 1e-10))
vh_db = clip(vh_db, -50, 10)
```

Do **not** gate conversion on per-patch max/min. Reference: `lulcdial/s1_vh_io.py::read_s1_vh_db`.

---

## 5. Scoring rules

1. Match predictions to bench rows by `patch_id`.
2. Normalize class strings: lowercase; split on `,` `/` `;` and the token ` and `.
3. Deduplicate class names while preserving order for display; sets are used for F1 / set-match.
4. Missing prediction for a bench row counts toward `missing_predictions` and is excluded from scored averages as implemented in `lulcdial.eval_zero_shot`.

**Dialogue metrics (dual):**

| Key | Meaning |
|-----|---------|
| `turn*_set_match_accuracy` | Exact set equality (strict; primary for dialogue “accuracy”) |
| `turn*_example_f1` | Soft: same example F1 as classify on that turn (secondary) |

On post-fix LULCDial GE preds, turn-1 **set-match ≈ 0.13** but turn-1 **example F1 ≈ 0.81** (≈ classify) — the model names the right classes conversationally but often fails exact-set equality.

Reference scorer:

```bash
python -m lulcdial.eval_zero_shot \
  --bench-jsonl ai4lcc_test.jsonl \
  --pred-jsonl predictions.jsonl \
  --out-metrics metrics.json
```

Prediction JSONL fields: `patch_id`, `pred_classify`, `pred_dialogue_turn1`, `pred_dialogue_turn2`.

---

## 6. Official reported baselines (post-radiometry-fix)

See [`leaderboard.csv`](leaderboard.csv). Summary:

| System | Split | N | Example F1 | T1 set | T2 set | T1 F1 | T2 F1 |
|--------|-------|---|------------|--------|--------|-------|-------|
| EarthDial_4B_MS (ZS) | GE test | 2497 | 0.052 | 0.000 | 0.000 | 0.000 | 0.000 |
| LULCDial_S1_v0.1 (FT) | GE test | 2497 | **0.812** | 0.134 | 0.390 | **0.813** | **0.870** |
| LULCDial_S1_v0.1 → MultiSenNA | transfer | 11939 | **0.679** | 0.013 | 0.079 | **0.687** | **0.686** |

Dialogue **improvement track:** [`LULCDial-s1/docs/DIALOGUE_IMPROVE.md`](../LULCDial-s1/docs/DIALOGUE_IMPROVE.md) (soft metrics done; format-aligned FT next).

---

## 7. What this bench is / is not

| Claim | Yes / No |
|-------|----------|
| SAR-VLM bench on official **14-class OCSGE** with classify + dialogue | Yes |
| New raster dataset release | No — protocol on AI4LCC patches |
| Multitemporal S1 stack | No (v0.1 = single date) |
| Pixel segmentation | No (patch multi-label) |
