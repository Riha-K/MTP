# LULCDial-S1 — AI4LCC data prep

> **Status (2026-07):** v0.1 bench (70/30, 2497 test) + ZS + FT **done**. Commands: root [`RUNBOOK.md`](../../RUNBOOK.md) · plan [`ROADMAP.md`](../../ROADMAP.md).

## What you download (official AI4LCC — **not** the HF tile clips)

| File | URL | Size | Required |
|------|-----|------|----------|
| Labels (JSON) | https://s3.unistra.fr/a2s_datasets/MultiSenGE/labels.tgz | ~4 MB | ✅ Already done |
| S1 patches | https://s3.unistra.fr/a2s_datasets/MultiSenGE/s1.tgz | ~110 GB | ✅ **You download** |
| Ground reference | https://s3.unistra.fr/a2s_datasets/MultiSenGE/ground_reference.tgz | ~25 MB | Optional |

**Do not use** `wtr001/S1_AI4LCC` huge tile `.tif` files for training — those are reprocessed mosaics, not 256×256 patches.

## Folder layout (after extract)

```text
LULCDial-s1/data/lulcdial_s1/ai4lcc/multisenge/
  labels/          ← 8,157 JSON files (done)
  s1/              ← extract s1.tgz here (many .tif per patch/date)
```

## Your steps

1. Download `s1.tgz` (~110 GB) to a drive with space.
2. Extract:
   ```powershell
   cd e:\MTP\earth2\LULCDial-s1\data\lulcdial_s1\ai4lcc\multisenge
   tar -xzf s1.tgz
   ```
   If the archive creates a nested folder, point `--s1-dir` at the folder that contains `*_S1_*.tif` files.

3. Install deps:
   ```powershell
   cd e:\MTP\earth2\LULCDial-s1
   python -m pip install -r lulcdial/requirements.txt
   ```

4. Tell me when extract is done (or the exact `--s1-dir` path). I will run:

   ```powershell
   cd e:\MTP\earth2\LULCDial-s1
   python -m lulcdial.build_instruct_s1 ^
     --labels-dir data/lulcdial_s1/ai4lcc/multisenge/labels ^
     --s1-dir data/lulcdial_s1/ai4lcc/multisenge/s1 ^
     --out-dir data/lulcdial_s1/shards/ai4lcc_ge_train ^
     --split all
   ```

## Output (EarthDial-ready)

| Output | Path | Contents |
|--------|------|----------|
| Train shard | `data/lulcdial_s1/shards/ai4lcc_ge_train_train/` | ~14.7k QA (8157 patches × 2 templates) |
| Val shard | `data/lulcdial_s1/shards/ai4lcc_ge_train_val/` | ~10% held-out |
| Bench JSONL | `data/lulcdial_s1/bench/v0.1/ai4lcc_test.jsonl` | 2497 test (70/30) |
| Stage 4 config | `src/shell/data/Stage4_LULCDial_S1.json` | Points EarthDial to shards |

Each training sample:
```python
{"jpg": PIL float32 VH dB 256×256, "conversations": '[{"from":"human",...},{"from":"gpt",...}]'}
```

## Zero-shot eval (Stage 1B)

1) Pack only val S1 TIFFs referenced by the bench (laptop / remote CPU):
```powershell
python -m lulcdial.pack_bench_s1 ^
  --bench-jsonl data/lulcdial_s1/bench/v0.1/ai4lcc_test.jsonl ^
  --src-s1-dir data/lulcdial_s1/ai4lcc/multisenge/s1 ^
  --out-dir data/lulcdial_s1/ai4lcc/multisenge/s1_test_bench_v0.1
```

2) On PARAM GPU — run EarthDial_4B_MS inference (full env pins in root `RUNBOOK.md`):
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
python -m lulcdial.predict_zero_shot \
  --bench-jsonl data/lulcdial_s1/bench/v0.1/ai4lcc_val.jsonl \
  --s1-root data/lulcdial_s1/ai4lcc/multisenge/s1_val_bench \
  --checkpoint /home/rihak_iitp/EarthDial_Models/EarthDial_4B_MS \
  --out-pred-jsonl data/lulcdial_s1/bench/v0.1/preds/earthdial_zs/ai4lcc_val_predictions.jsonl
```

3) Score:
```powershell
python -m lulcdial.eval_zero_shot ^
  --bench-jsonl data/lulcdial_s1/bench/v0.1/ai4lcc_val.jsonl ^
  --pred-jsonl data/lulcdial_s1/bench/v0.1/preds/earthdial_zs/ai4lcc_val_predictions.jsonl ^
  --out-metrics data/lulcdial_s1/metrics/v0.1/earthdial_zs_baseline.json
```

**v0.1 DONE** (ZS F1 ≈ 0.019 → FT **0.800** on 2497 test).  
Metrics: `data/lulcdial_s1/metrics/v0.1/`.  
**Next:** MultiSenNA transfer with `LULCDial_S1_v0.1` (70/30 checkpoint; never train on NA).

## MultiSenNA prep (Stage 2 transfer eval) — NEXT

Bench JSONL (~12k) is already on PARAM (`bench/v0.1/multisenna_bench.jsonl` or under `bench/multisenna/`). **Do not train on NA.**

Folders if you need to rebuild:

```text
data/lulcdial_s1/ai4lcc/multisenna/
  labels/            ← extract MultiSenNA labels JSON here
  s1/                ← extract MultiSenNA S1 tif here

data/lulcdial_s1/bench/multisenna/v0.1/
  multisenna_bench.jsonl
  multisenna_bench.summary.json
```

Build MultiSenNA bench JSONL (all patches by default):

```powershell
cd e:\MTP\earth2\LULCDial-s1
python -m lulcdial.multisenna.build_bench_multisenna ^
  --labels-dir data/lulcdial_s1/ai4lcc/multisenna/labels ^
  --s1-dir data/lulcdial_s1/ai4lcc/multisenna/s1 ^
  --out-jsonl data/lulcdial_s1/bench/multisenna/v0.1/multisenna_bench.jsonl
```

Optional smoke test first:

```powershell
python -m lulcdial.multisenna.build_bench_multisenna ^
  --labels-dir data/lulcdial_s1/ai4lcc/multisenna/labels ^
  --s1-dir data/lulcdial_s1/ai4lcc/multisenna/s1 ^
  --out-jsonl data/lulcdial_s1/bench/multisenna/v0.1/multisenna_bench_smoke.jsonl ^
  --max-samples 100
```

## Quick test before full 110 GB download

Extract only `labels.tgz` (done) and run a dry check:
```powershell
python -c "from lulcdial.patch_meta import iter_patches; p=iter_patches('data/lulcdial_s1/ai4lcc/multisenge/labels'); print(len(p), p[0])"
```
