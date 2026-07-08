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

## Quick test before full 110 GB download

Extract only `labels.tgz` (done) and run a dry check:
```powershell
python -c "from baresoil.patch_meta import iter_patches; p=iter_patches('data/baresoil_s1/ai4lcc/multisenge/labels'); print(len(p), p[0])"
```
