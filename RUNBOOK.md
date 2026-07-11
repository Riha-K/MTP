# LULCDial-S1 — Runbook (copy-paste commands)

> **One file for the full pipeline:** data → bench → zero-shot → fine-tune → eval → MultiSenNA  
> **Workspace:** `e:\MTP\earth2\`  
> **Code root:** `LULCDial-s1\`  
> **Status (2026-07-11):** **1A done** · **1B done** (ZS F1 ≈ 0.0194 on 801) · **next = 1C-a (25% FT)**

---

## Which file holds what


| File | What it is |
|------|------------|
| **`RUNBOOK.md`** (this file) | **All commands** — 1A/1B/1C/1D + **PARAM GPU env pins** |
| **`log.md`** | What changed / when / results (newest first) |
| **`README.md`** | Reading order + current status pointer |
| **`Stage1_Summer_Intern_Guide.md`** | Stage goals, glossary, timeline |
| **`AI4LCC_S1_VLM_MTech_3Stage_Roadmap.md`** | Full thesis plan (3 stages) |
| **`LULCDial-s1/baresoil/README.md`** | Data-prep folder layout + short commands |
| **`LULCDial-s1/src/shell/data/Stage4_BareSoil_S1.json`** | Fine-tune shard paths (edit before 1C) |

---

## Where to run what


| Machine             | Use for                                                       |
| ------------------- | ------------------------------------------------------------- |
| **Computer system** | Extract raw `s1.tgz`, build shards + bench (disk + RAM heavy) |
| **PARAM GPU**       | Zero-shot inference, fine-tune, final eval (GPU heavy)        |
| **Laptop**          | Code, docs, smoke tests (`--max-patches`), git, monitoring    |


**Rule:** After shards + bench are built, copy only prepared artifacts to GPU server — not full 110 GB raw S1 unless needed.

---

## Status checklist (live)


| Stage | Status | Artifact |
|-------|--------|----------|
| **1A** shards + GE bench | **DONE** | train 14710 / val 1602 shards; `bench/v0.1/ai4lcc_val.jsonl` (801) |
| **1A** MultiSenNA bench | **DONE** | `bench/v0.1/multisenna_bench.jsonl` (~12k) on PARAM |
| **1B** EarthDial ZS | **DONE** | `metrics/v0.1/earthdial_zs_baseline.json` (F1 ≈ 0.0194) |
| **1C-a** 25% fine-tune | **NEXT** | subsample train shard → FT → `metrics/v0.1/lulcdial_p25.json` |
| **1C-b / 1C-c** 50% / 100% | pending | same pattern |
| **1D** beat ZS | after each 1C | compare metrics to ZS baseline |

---



## Key paths (always the same)

```text
e:\MTP\earth2\
├── RUNBOOK.md                          ← this file (commands + PARAM env)
├── Stage1_Summer_Intern_Guide.md       ← stage goals + glossary
├── log.md                              ← change history
└── LULCDial-s1\
    ├── baresoil\                       ← data-prep + eval scripts
    │   ├── build_instruct_s1.py        ← shards (train/val)
    │   ├── build_bench.py              ← MultiSenGE val bench
    │   ├── pack_bench_s1.py            ← pack ~801 val TIFFs for PARAM
    │   ├── predict_zero_shot.py        ← EarthDial / FT inference
    │   ├── eval_zero_shot.py           ← score predictions
    │   └── multisenna\
    │       └── build_bench_multisenna.py
    ├── data\baresoil_s1\
    │   ├── ai4lcc\multisenge\labels\   ← 8,157 JSON
    │   ├── ai4lcc\multisenge\s1\       ← full S1 (remote CPU only)
    │   ├── ai4lcc\multisenge\s1_val_bench\  ← ~801 TIFFs on PARAM
    │   ├── shards\                     ← HF train/val (+ p25/p50 later)
    │   ├── bench\v0.1\
    │   │   ├── ai4lcc_val.jsonl
    │   │   └── preds\earthdial_zs\…   ← ZS preds (committed)
    │   └── metrics\v0.1\               ← earthdial_zs_baseline.json
    ├── src\earthdial\train\finetune.py
    └── src\shell\data\Stage4_BareSoil_S1.json
```

---

## PARAM GPU env (locked 2026-07-10 — reuse for 1C)

**Login:** `ssh rihak_iitp@paramrudra.iitp.ac.in` (CAPTCHA + password)  
**Code:** `~/MTP/earth2/LULCDial-s1`  
**Model:** `~/EarthDial_Models/EarthDial_4B_MS` (~8.3 GB; RGB-only is **not** enough)  
**Already on PARAM:** full train/val shards, `s1_val_bench` (801), GE + MultiSenNA benches

### Every GPU session (do in this order)

```bash
# on login01
salloc -N 1 -n 4 -p gpu --gres=gpu:1 -t 04:00:00
srun --pty bash          # MUST — else you stay on login01
hostname                 # expect ragpu0XX

module purge
module load MLDL/Pytorch-gpu   # capital MLDL — not mldl/

# long jobs: protect from SSH drop
tmux new -s ft25         # or: tmux attach -t ft25
```

### Python pins that made ZS work (install once per user; re-check after module load)

```bash
# EarthDial-compatible stack (do NOT let peft/transformers float to latest)
pip install --user "transformers==4.37.2" "tokenizers==0.15.1" "peft==0.10.0"
pip install --user "numpy==1.26.4" protobuf sentencepiece
pip install --user datasets rasterio tqdm Pillow tifffile

# Avoid: opencv-python-headless (pulled numpy 2.x and broke torch)
# Optional noise only: FlashAttention missing — OK to ignore for ZS/FT
```

**Sanity:**

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# expect: 2.2.x  True
```

### Download MS weights (already done; keep for reference)

```bash
# on login01 (no GPU needed)
mkdir -p ~/EarthDial_Models
huggingface-cli download akshaydudhane/EarthDial_4B_MS \
  --local-dir ~/EarthDial_Models/EarthDial_4B_MS
```

### Ops notes

- `git` often missing on **compute** nodes — `git pull` on **login01**, then enter GPU.
- SSH drop kills non-tmux jobs — always use **tmux** for full 801 / fine-tune.
- Predict supports `--resume` (skips rows already in the pred JSONL).

---



## Step 0 — Setup (any machine)

```powershell
cd e:\MTP\earth2
git pull

cd LULCDial-s1
python -m pip install -r baresoil/requirements.txt
```

**Quick label check (no S1 needed):**

```powershell
cd e:\MTP\earth2\LULCDial-s1
python -c "from baresoil.patch_meta import iter_patches; p=iter_patches('data/baresoil_s1/ai4lcc/multisenge/labels'); print(len(p), p[0].stem)"
```

Expected: `8157` patches.

---



## Step 1A — Build training data (MultiSenGE)



### 1A.0 Download + extract (professor system recommended)


| File       | URL                                                                                                                    |
| ---------- | ---------------------------------------------------------------------------------------------------------------------- |
| S1 patches | [https://s3.unistra.fr/a2s_datasets/MultiSenGE/s1.tgz](https://s3.unistra.fr/a2s_datasets/MultiSenGE/s1.tgz) (~110 GB) |


```powershell
cd e:\MTP\earth2\LULCDial-s1\data\baresoil_s1\ai4lcc\multisenge
tar -xzf s1.tgz
```

Point `--s1-dir` at the folder that contains `*_S1_*.tif` files.

### 1A.1 Smoke test first (laptop or server)

```powershell
cd e:\MTP\earth2\LULCDial-s1

python -m baresoil.build_instruct_s1 ^
  --labels-dir data/baresoil_s1/ai4lcc/multisenge/labels ^
  --s1-dir data/baresoil_s1/ai4lcc/multisenge/s1 ^
  --out-dir data/baresoil_s1/shards/ai4lcc_ge_train_smoke ^
  --split all ^
  --max-patches 20

python -m baresoil.build_bench ^
  --labels-dir data/baresoil_s1/ai4lcc/multisenge/labels ^
  --s1-dir data/baresoil_s1/ai4lcc/multisenge/s1 ^
  --out-jsonl data/baresoil_s1/bench/v0.1/ai4lcc_val_smoke.jsonl ^
  --max-samples 20
```



### 1A.2 Full build (professor system)

```powershell
cd e:\MTP\earth2\LULCDial-s1

python -m baresoil.build_instruct_s1 ^
  --labels-dir data/baresoil_s1/ai4lcc/multisenge/labels ^
  --s1-dir data/baresoil_s1/ai4lcc/multisenge/s1 ^
  --out-dir data/baresoil_s1/shards/ai4lcc_ge_train ^
  --split all

python -m baresoil.build_bench ^
  --labels-dir data/baresoil_s1/ai4lcc/multisenge/labels ^
  --s1-dir data/baresoil_s1/ai4lcc/multisenge/s1 ^
  --out-jsonl data/baresoil_s1/bench/v0.1/ai4lcc_val.jsonl
```



### 1A.3 Verify outputs


| Artifact    | Path                                             | Done when                           |
| ----------- | ------------------------------------------------ | ----------------------------------- |
| Train shard | `data/baresoil_s1/shards/ai4lcc_ge_train_train/` | `manifest.json` → `num_samples > 0` |
| Val shard   | `data/baresoil_s1/shards/ai4lcc_ge_train_val/`   | same                                |
| Bench JSONL | `data/baresoil_s1/bench/v0.1/ai4lcc_val.jsonl`   | ~800 rows                           |


**Copy to GPU server (minimum):**

- `shards/ai4lcc_ge_train_train/`
- `shards/ai4lcc_ge_train_val/`
- `bench/v0.1/ai4lcc_val.jsonl`

---



## Step 1B — Zero-shot baseline (before fine-tune)

**Model:** `EarthDial_4B_MS` (no AI4LCC fine-tune yet)  
**Eval data:** `bench/v0.1/ai4lcc_val.jsonl` (801 rows)  
**Need on GPU server:** model weights + **val S1 TIFFs** (not full 110 GB — only files listed in the bench)

### 1B.0 Pack + upload val S1 only (~801 files)

On laptop / remote CPU (where full `multisenge/s1` exists):

```powershell
cd e:\MTP\earth2\LULCDial-s1   # or D:\Riha\earth2\LULCDial-s1

python -m baresoil.pack_bench_s1 ^
  --bench-jsonl data/baresoil_s1/bench/v0.1/ai4lcc_val.jsonl ^
  --src-s1-dir data/baresoil_s1/ai4lcc/multisenge/s1 ^
  --out-dir data/baresoil_s1/ai4lcc/multisenge/s1_val_bench
```

Then `scp -r` that `s1_val_bench` folder to PARAM:

```text
~/MTP/earth2/LULCDial-s1/data/baresoil_s1/ai4lcc/multisenge/s1_val_bench/
```

Also ensure `EarthDial_4B_MS` exists under `~/EarthDial_Models/` (RGB-only is not enough for S1).

### 1B.1 Smoke inference (20 samples) then full

On PARAM **GPU node** (not login-only if GPU is unavailable there):

```bash
cd ~/MTP/earth2/LULCDial-s1
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# smoke
python -m baresoil.predict_zero_shot \
  --bench-jsonl data/baresoil_s1/bench/v0.1/ai4lcc_val.jsonl \
  --s1-root data/baresoil_s1/ai4lcc/multisenge/s1_val_bench \
  --checkpoint /home/rihak_iitp/EarthDial_Models/EarthDial_4B_MS \
  --out-pred-jsonl data/baresoil_s1/bench/v0.1/preds/earthdial_zs/ai4lcc_val_predictions.jsonl \
  --max-samples 20

# full (resume-safe) — ~47 min for 801 on A100
python -m baresoil.predict_zero_shot \
  --bench-jsonl data/baresoil_s1/bench/v0.1/ai4lcc_val.jsonl \
  --s1-root data/baresoil_s1/ai4lcc/multisenge/s1_val_bench \
  --checkpoint /home/rihak_iitp/EarthDial_Models/EarthDial_4B_MS \
  --out-pred-jsonl data/baresoil_s1/bench/v0.1/preds/earthdial_zs/ai4lcc_val_predictions.jsonl \
  --resume
```

### 1B.2 Score zero-shot

```bash
python -m baresoil.eval_zero_shot \
  --bench-jsonl data/baresoil_s1/bench/v0.1/ai4lcc_val.jsonl \
  --pred-jsonl data/baresoil_s1/bench/v0.1/preds/earthdial_zs/ai4lcc_val_predictions.jsonl \
  --out-metrics data/baresoil_s1/metrics/v0.1/earthdial_zs_baseline.json
```

**Done when:** `metrics/v0.1/earthdial_zs_baseline.json` exists (committed; F1 ≈ 0.0194).

**Notes (do not redesign mid-1B):**
- Keep GT answers as **clean class lists** (e.g. `Arable Lands, Grasslands, Forests`).
- Strict example-F1 / set-match is the **primary** metric. Low ZS F1 is expected and useful.
- Do **not** change answer templates or metrics until 1B baseline is saved.

---



## Step 1C — Data-scaling fine-tune (25% → 50% → 100%)

**Goal:** Prove the model **gains knowledge with more AI4LCC data** (not that a tiny subset already saturates).

**Important:** The train folder’s 8 `.arrow` files are **one** dataset (~14710 samples), not 8 separate corpora. Scaling = **% of train patches / samples**, not “use 2 arrow files only.”

**Rules:**
- Same base every time: `EarthDial_4B_MS` (separate short runs — **not** one long continued train)
- Same hyperparams across 25 / 50 / 100%
- Same bench: `ai4lcc_val.jsonl` (801) + same `eval_zero_shot` scorer
- **Do not** use “2 of 8 `.arrow` files” as 25% — subsample **rows** from the uploaded full train shard

| Run | Train size (approx) | Checkpoint out | Metrics out |
|-----|---------------------|----------------|-------------|
| **1C-a** | ~25% train samples | `checkpoints/LULCDial_S1_p25/` | `metrics/v0.1/lulcdial_p25.json` |
| **1C-b** | ~50% train samples | `checkpoints/LULCDial_S1_p50/` | `metrics/v0.1/lulcdial_p50.json` |
| **1C-c** | **100%** (full shard) | `checkpoints/LULCDial_S1_v0.1/` | `metrics/v0.1/lulcdial_v0.1.json` |

### 1C.0 Subsample train shard on PARAM (no raw S1 needed)

Full train shard is already on PARAM. Keep it for 100%; write a new folder for 25%:

```bash
cd ~/MTP/earth2/LULCDial-s1
python - <<'PY'
from datasets import load_from_disk
src = "data/baresoil_s1/shards/ai4lcc_ge_train_train"
dst = "data/baresoil_s1/shards/ai4lcc_ge_train_p25"
ds = load_from_disk(src)
n = int(round(len(ds) * 0.25))  # ~3678 of 14710 (2 QA per patch)
print(f"full={len(ds)} p25={n}")
ds.select(range(n)).save_to_disk(dst)
print("wrote", dst)
PY
```

Then point `Stage4_BareSoil_S1.json` train `annotation` → `.../ai4lcc_ge_train_p25` (PARAM Linux path). Val stays `ai4lcc_ge_train_val`.

**Interpretation:**
- If **ZS ≪ 25% ≪ 100%** → genuine learning (good thesis plot)
- If **25% ≈ 100%** → scaling weak; debug LR / epochs / templates before claiming full-data win

**Optional later (after scaling — pick once):**
- Light sentence wrappers around the **same** class names + scorer tweak
- Relaxed F1 with aliases — report **beside** strict F1, never instead

---



## Step 1C (legacy block) — Fine-tune LULCDial-S1 v0.1 (GPU server)

> Prefer the **25 → 50 → 100%** plan above. Full 100% run is **1C-c**.

**Config:** `LULCDial-s1/src/shell/data/Stage4_BareSoil_S1.json`  
**Trainer:** `LULCDial-s1/src/earthdial/train/finetune.py`  
**Base checkpoint:** `EarthDial_4B_MS`  
**Output:** `LULCDial-s1/checkpoints/LULCDial_S1_v0.1/`

Before training, update shard paths inside `Stage4_BareSoil_S1.json` if machine path differs from:

```text
e:/MTP/earth2/LULCDial-s1/data/baresoil_s1/shards/ai4lcc_ge_train_train
e:/MTP/earth2/LULCDial-s1/data/baresoil_s1/shards/ai4lcc_ge_train_val
```

**Template command (Linux GPU server — adapt paths/GPUs):**

```bash
cd /path/to/LULCDial-s1/src
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

torchrun \
  --nnodes=1 \
  --nproc_per_node=8 \
  --master_port=34229 \
  earthdial/train/finetune.py \
  --model_name_or_path "/path/to/EarthDial_4B_MS" \
  --conv_style "phi3-chat" \
  --output_dir "/path/to/checkpoints/LULCDial_S1_v0.1" \
  --meta_path "shell/data/Stage4_BareSoil_S1.json" \
  --overwrite_output_dir True \
  --force_image_size 224 \
  --bf16 True \
  --num_train_epochs 1 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 64 \
  --learning_rate 4e-5 \
  --do_train True \
  --deepspeed "shell/zero_stage1_config.json"
```

See also: `LULCDial-s1/src/EarthDial.sh` (example shell wrapper).

---



## Step 1D — Eval fine-tuned model (same bench as 1B)

1. Run inference with **LULCDial-S1 v0.1** checkpoint on same `ai4lcc_val.jsonl`.
2. Save as e.g. `ai4lcc_val_predictions_lulcdial.jsonl`.
3. Score with same script:

```powershell
cd e:\MTP\earth2\LULCDial-s1

python -m baresoil.eval_zero_shot ^
  --bench-jsonl data/baresoil_s1/bench/v0.1/ai4lcc_val.jsonl ^
  --pred-jsonl data/baresoil_s1/bench/v0.1/ai4lcc_val_predictions_lulcdial.jsonl ^
  --out-metrics data/baresoil_s1/metrics/lulcdial_v0.1.json
```

**Compare:**


| File                                 | Model                            |
| ------------------------------------ | -------------------------------- |
| `metrics/earthdial_zs_baseline.json` | EarthDial ZS (1B)                |
| `metrics/lulcdial_v0.1.json`         | LULCDial-S1 after fine-tune (1D) |


---



## Step 2 — MultiSenNA prep (transfer eval, later)

**Extract data here:**

```text
LULCDial-s1/data/baresoil_s1/ai4lcc/multisenna/labels/
LULCDial-s1/data/baresoil_s1/ai4lcc/multisenna/s1/
```

**Smoke test:**

```powershell
cd e:\MTP\earth2\LULCDial-s1

python -m baresoil.multisenna.build_bench_multisenna ^
  --labels-dir data/baresoil_s1/ai4lcc/multisenna/labels ^
  --s1-dir data/baresoil_s1/ai4lcc/multisenna/s1 ^
  --out-jsonl data/baresoil_s1/bench/multisenna/v0.1/multisenna_bench_smoke.jsonl ^
  --max-samples 100
```

**Full bench:**

```powershell
python -m baresoil.multisenna.build_bench_multisenna ^
  --labels-dir data/baresoil_s1/ai4lcc/multisenna/labels ^
  --s1-dir data/baresoil_s1/ai4lcc/multisenna/s1 ^
  --out-jsonl data/baresoil_s1/bench/multisenna/v0.1/multisenna_bench.jsonl
```

**Outputs:**

- `bench/multisenna/v0.1/multisenna_bench.jsonl`
- `bench/multisenna/v0.1/multisenna_bench.summary.json`

Use same `eval_zero_shot.py` flow for scoring (never train on MultiSenNA).

---



## Stage checklist (quick)


| Stage  | Command module                            | Exit artifact                   |
| ------ | ----------------------------------------- | ------------------------------- |
| **1A** | `build_instruct_s1`, `build_bench`        | shards + `ai4lcc_val.jsonl` — **DONE** |
| **1B** | `predict_zero_shot` + `eval_zero_shot`    | `metrics/v0.1/earthdial_zs_baseline.json` — **DONE** |
| **1C** | fine-tune 25% → 50% → 100% (separate runs from `EarthDial_4B_MS`) | `LULCDial_S1_p25/p50/v0.1` — **NEXT: p25** |
| **1D** | `eval_zero_shot` after each 1C run        | `metrics/v0.1/lulcdial_p25/p50/v0.1.json` |
| **2**  | `build_bench_multisenna`                  | `multisenna_bench.jsonl`        |


---

## PARAM demo startup (optional, server only)

Use the updated demo launcher with env overrides:

```bash
cd ~/MTP/earth2/LULCDial-s1/demo
export EARTHDIAL_GPU=1
export EARTHDIAL_MODEL_PATH=/home/rihak_iitp/EarthDial_Models/EarthDial_4B_MS
export EARTHDIAL_CONTROLLER_URL=http://0.0.0.0:40000
bash earthdial_demo.sh
```

`demo/app.py` also reads `EARTHDIAL_CONTROLLER_URL` / `EARTHDIAL_SD_WORKER_URL` if you start Streamlit manually.

---

## Related docs (deeper reading)


| File                                                                                                                 | Purpose                            |
| -------------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| [`Stage1_Summer_Intern_Guide.md`](Stage1_Summer_Intern_Guide.md)                                                     | Stage goals, glossary, timeline    |
| [`LULCDial-s1/baresoil/README.md`](LULCDial-s1/baresoil/README.md)                                                   | Data-prep commands + folder layout |
| [`BenchmarkGuide/AI4LCC/BareSoil_AI4LCC_Workflow_Guide.md`](BenchmarkGuide/AI4LCC/BareSoil_AI4LCC_Workflow_Guide.md) | PI-level workflow explanation      |
| [`AI4LCC_S1_VLM_MTech_3Stage_Roadmap.md`](AI4LCC_S1_VLM_MTech_3Stage_Roadmap.md)                                     | Full thesis roadmap (3 stages)     |
| [`log.md`](log.md)                                                                                                   | What changed in repo and when      |


