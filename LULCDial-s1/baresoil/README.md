# BareSoil S1 — AI4LCC data prep

## What you download (official AI4LCC — **not** the HF tile clips)

| File | URL | Size | Required |
|------|-----|------|----------|
| Labels (JSON) | https://s3.unistra.fr/a2s_datasets/MultiSenGE/labels.tgz | ~4 MB | ✅ Already done |
| S1 patches | https://s3.unistra.fr/a2s_datasets/MultiSenGE/s1.tgz | ~110 GB | ✅ **You download** |
| Ground reference | https://s3.unistra.fr/a2s_datasets/MultiSenGE/ground_reference.tgz | ~25 MB | Optional |

**Do not use** `wtr001/S1_AI4LCC` huge tile `.tif` files for training — those are reprocessed mosaics, not 256×256 patches.

## Folder layout (after extract)

```text
LULCDial-s1/data/baresoil_s1/ai4lcc/multisenge/
  labels/          ← 8,157 JSON files (done)
  s1/              ← extract s1.tgz here (many .tif per patch/date)
```

## Your steps

1. Download `s1.tgz` (~110 GB) to a drive with space.
2. Extract:
   ```powershell
   cd e:\MTP\earth2\LULCDial-s1\data\baresoil_s1\ai4lcc\multisenge
   tar -xzf s1.tgz
   ```
   If the archive creates a nested folder, point `--s1-dir` at the folder that contains `*_S1_*.tif` files.

3. Install deps:
   ```powershell
   cd e:\MTP\earth2\LULCDial-s1
   python -m pip install -r baresoil/requirements.txt
   ```

4. Tell me when extract is done (or the exact `--s1-dir` path). I will run:

   ```powershell
   cd e:\MTP\earth2\LULCDial-s1
   python -m baresoil.build_instruct_s1 ^
     --labels-dir data/baresoil_s1/ai4lcc/multisenge/labels ^
     --s1-dir data/baresoil_s1/ai4lcc/multisenge/s1 ^
     --out-dir data/baresoil_s1/shards/ai4lcc_ge_train ^
     --split all
   ```

## Output (EarthDial-ready)

| Output | Path | Contents |
|--------|------|----------|
| Train shard | `data/baresoil_s1/shards/ai4lcc_ge_train_train/` | ~14.7k QA (8157 patches × 2 templates) |
| Val shard | `data/baresoil_s1/shards/ai4lcc_ge_train_val/` | ~10% held-out |
| Bench JSONL | `data/baresoil_s1/bench/v0.1/ai4lcc_val.jsonl` | Classify + 2-turn dialogue |
| Stage 4 config | `src/shell/data/Stage4_BareSoil_S1.json` | Points EarthDial to shards |

Each training sample:
```python
{"jpg": PIL float32 VH dB 256×256, "conversations": '[{"from":"human",...},{"from":"gpt",...}]'}
```

## Zero-shot eval prep (Stage 1B only)

Use this before fine-tuning to keep a clean baseline workflow.

1) Export eval requests from bench:
```powershell
cd e:\MTP\earth2\LULCDial-s1
python -m baresoil.eval_zero_shot ^
  --bench-jsonl data/baresoil_s1/bench/v0.1/ai4lcc_val.jsonl ^
  --out-metrics data/baresoil_s1/metrics/_tmp.json ^
  --dump-requests-jsonl data/baresoil_s1/bench/v0.1/ai4lcc_val_requests.jsonl
```

2) Run model inference separately and save prediction rows JSONL with:
- `patch_id`
- `pred_classify`
- `pred_dialogue_turn1`
- `pred_dialogue_turn2`

3) Score predictions:
```powershell
python -m baresoil.eval_zero_shot ^
  --bench-jsonl data/baresoil_s1/bench/v0.1/ai4lcc_val.jsonl ^
  --pred-jsonl data/baresoil_s1/bench/v0.1/ai4lcc_val_predictions.jsonl ^
  --out-metrics data/baresoil_s1/metrics/earthdial_zs_baseline.json
```

This path is eval-only and does not modify Stage 4 fine-tune config.

## MultiSenNA prep (Stage 2 transfer eval)

Use these folders now so data can be dropped/extracted once and kept ready:

```text
data/baresoil_s1/ai4lcc/multisenna/
  labels/            ← extract MultiSenNA labels JSON here
  s1/                ← extract MultiSenNA S1 tif here

data/baresoil_s1/bench/multisenna/v0.1/
  multisenna_bench.jsonl
  multisenna_bench.summary.json
```

Build MultiSenNA bench JSONL (all patches by default):

```powershell
cd e:\MTP\earth2\LULCDial-s1
python -m baresoil.multisenna.build_bench_multisenna ^
  --labels-dir data/baresoil_s1/ai4lcc/multisenna/labels ^
  --s1-dir data/baresoil_s1/ai4lcc/multisenna/s1 ^
  --out-jsonl data/baresoil_s1/bench/multisenna/v0.1/multisenna_bench.jsonl
```

Optional smoke test first:

```powershell
python -m baresoil.multisenna.build_bench_multisenna ^
  --labels-dir data/baresoil_s1/ai4lcc/multisenna/labels ^
  --s1-dir data/baresoil_s1/ai4lcc/multisenna/s1 ^
  --out-jsonl data/baresoil_s1/bench/multisenna/v0.1/multisenna_bench_smoke.jsonl ^
  --max-samples 100
```

## Quick test before full 110 GB download

Extract only `labels.tgz` (done) and run a dry check:
```powershell
python -c "from baresoil.patch_meta import iter_patches; p=iter_patches('data/baresoil_s1/ai4lcc/multisenge/labels'); print(len(p), p[0])"
```
