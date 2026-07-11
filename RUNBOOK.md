# LULCDial-S1 — Runbook (copy-paste commands)

> **One file for the full pipeline:** data → bench → zero-shot → fine-tune → eval → MultiSenNA  
> **Workspace:** `e:\MTP\earth2\`  
> **Code root:** `LULCDial-s1\`  
> **Status (2026-07-11):** **1A done** · **1B done** (ZS F1 ≈ 0.0194 on 801) · **1C-a RUNNING** (`sbatch` job — prefer this over interactive)

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
| **1C-a** 25% fine-tune | **RUNNING (sbatch)** | job `ft25` → `checkpoints/LULCDial_S1_p25/` (~60–80 min, 41 steps) |
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

### Prefer `sbatch` for training (survives SSH / closing CMD)

Interactive `salloc` + `tmux` **failed repeatedly**: SSH reset or exiting the outer `srun` shell kills the allocation (and tmux with it). Use **`sbatch`** for 1C fine-tunes.

```bash
# on login01 — create once, then sbatch
cat > ~/train_p25.sh << 'EOF'
#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 04:00:00
#SBATCH -J ft25
# IMPORTANT: use absolute paths — Slurm does NOT expand ~
#SBATCH -o /home/rihak_iitp/ft25_%j.out
#SBATCH -e /home/rihak_iitp/ft25_%j.err

module purge
module load MLDL/Pytorch-gpu
cd /home/rihak_iitp/MTP/earth2/LULCDial-s1/src
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m torch.distributed.run \
  --nnodes=1 --nproc_per_node=1 --master_port=34229 \
  earthdial/train/finetune.py \
  --model_name_or_path /home/rihak_iitp/EarthDial_Models/EarthDial_4B_MS \
  --conv_style "phi3-chat" \
  --output_dir /home/rihak_iitp/MTP/earth2/LULCDial-s1/checkpoints/LULCDial_S1_p25 \
  --meta_path shell/data/Stage4_BareSoil_S1.json \
  --overwrite_output_dir True \
  --force_image_size 448 --bf16 True --num_train_epochs 1 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 128 \
  --grad_checkpoint True --freeze_backbone True --max_seq_length 1024 \
  --evaluation_strategy "no" --save_strategy "epoch" --logging_steps 1 \
  --learning_rate 4e-5 --do_train True
EOF

chmod +x ~/train_p25.sh
sbatch ~/train_p25.sh
squeue -u $USER
```

**Monitor (safe to close CMD anytime):**

```bash
squeue -u $USER
tail -f ~/ft25_<JOBID>.out          # after fixing #SBATCH paths
# if an older script used ~/ in #SBATCH, logs landed in literal ~/~/ :
#   tail -f ~/~/ft25_<JOBID>.out
grep loss ~/ft25_<JOBID>.out | tail -5
ls -lt ~/MTP/earth2/LULCDial-s1/checkpoints/LULCDial_S1_p25/
```

Done when `squeue` empty **and** checkpoint has model files (not only `runs/`).

### Interactive GPU session (debug / short jobs only)

```bash
# on login01
salloc -N 1 -n 4 -p gpu --gres=gpu:1 -t 04:00:00
srun --pty bash          # MUST — else you stay on login01
hostname                 # expect racn1XX / ragpu0XX

module purge
module load MLDL/Pytorch-gpu   # capital MLDL — not mldl/
# prompt must show (Pytorch-gpu) — do NOT run conda deactivate (undoes the module)
```

### Python pins (install on **login01** with `MLDL/Pytorch-gpu` loaded — Python **3.10**)

```bash
# MUST see Pytorch-gpu + 3.10 before any pip --user
module purge
module load MLDL/Pytorch-gpu
which python   # .../envs/Pytorch-gpu/bin/python
python -c "import sys; print(sys.version)"   # 3.10.x

# EarthDial-compatible stack (do NOT let peft/transformers float to latest)
pip install --user "transformers==4.37.2" "tokenizers==0.15.1" "peft==0.10.0"
pip install --user "numpy==1.26.4" protobuf sentencepiece
pip install --user deepspeed==0.13.5 einops einops-exts timm==0.9.12
pip install --user datasets imageio orjson shortuuid termcolor yacs tensorboardX
pip install --user opencv-python-headless decord
pip install --user "numpy==1.26.4"   # re-pin after opencv (it wants numpy 2.x)

# NEVER: pip install in plain (base) / Python 3.13 — those packages are invisible to Pytorch-gpu
# FlashAttention missing — OK; code falls back to eager attention
```

**Sanity (on GPU node after module load):**

```bash
python -c "import torch, deepspeed, cv2, imageio, decord; print(torch.__version__, torch.cuda.is_available())"
# expect: 2.2.x  True
```

### Download MS weights (already done; keep for reference)

```bash
# on login01 (no GPU needed)
mkdir -p ~/EarthDial_Models
huggingface-cli download akshaydudhane/EarthDial_4B_MS \
  --local-dir ~/EarthDial_Models/EarthDial_4B_MS
```

### Ops notes (hard-won)

- Prefer **`sbatch`** for fine-tune — closing CMD / SSH drop does not kill the job.
- Interactive `salloc`+`srun`+`tmux`: detach (`Ctrl+B` then `D`) is **not enough** if you then `exit` the outer `srun` shell or that SSH session dies — Slurm cancels the whole allocation.
- `#SBATCH -o ~/…` does **not** expand `~` → logs go to literal `~/~/ft25_JOBID.out`. Use `/home/rihak_iitp/…`.
- `git` missing on **compute** nodes — `git pull` only on **login01** (home is shared).
- Prompt must say `(Pytorch-gpu)` before any `python` train command — not `(base)` / py3.13.
- **Never** `conda deactivate` after `module load MLDL/Pytorch-gpu` — it drops you back to base without torch.
- Do **not** nest `salloc` inside an existing job; cancel extras with `scancel`.
- Compute nodes often have **no internet** — `pip` only on login01.
- Fine-tune needs distributed launcher: use `python -m torch.distributed.run` (plain `python finetune.py` → `KeyError: RANK`).
- DeepSpeed may JIT-compile ops and fail with `which c++` missing — on **1× A100 80GB**, omit `--deepspeed` (bf16 is enough for 4B).
- Predict supports `--resume` (skips rows already in the pred JSONL).
- Aborted mid-epoch runs leave only `runs/` (TensorBoard) — `save_strategy epoch` means **no weights** until epoch ends. Restart is a clean load of `EarthDial_4B_MS`.

### 1C-a flags (must-haves)

**Important:** use `--force_image_size 448` (not 224). S1 `sequential_vit_features` emits 256 tokens; 224 makes the text side expect 64 and zeroes the loss. Match ZS / EarthDial S1.

**ETA (p25, 41 steps):** about **60–80 min** wall (~60–100 s/step). First step slowest. Healthy first `loss` ≈ **2.4** (not ~0).

`Stage4_BareSoil_S1.json` train path must be `.../shards/ai4lcc_ge_train_p25`.

Full launch command: see **Prefer `sbatch`** block above (same flags as the interactive copy used earlier).

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

**Template command (PARAM — 1 GPU; do NOT use plain `python finetune.py`):**

```bash
cd ~/MTP/earth2/LULCDial-s1/src
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=0

python -m torch.distributed.run \
  --nnodes=1 \
  --nproc_per_node=1 \
  --master_port=34229 \
  earthdial/train/finetune.py \
  --model_name_or_path /home/rihak_iitp/EarthDial_Models/EarthDial_4B_MS \
  --conv_style "phi3-chat" \
  --output_dir /home/rihak_iitp/MTP/earth2/LULCDial-s1/checkpoints/LULCDial_S1_p25 \
  --meta_path shell/data/Stage4_BareSoil_S1.json \
  --overwrite_output_dir True \
  --force_image_size 224 \
  --bf16 True \
  --num_train_epochs 1 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 64 \
  --learning_rate 4e-5 \
  --do_train True
```

For 8-GPU machines, set `--nproc_per_node=8`, drop `CUDA_VISIBLE_DEVICES`, and you may add `--deepspeed shell/zero_stage1_config.json` if `g++` is available.

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


