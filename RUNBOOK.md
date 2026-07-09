# LULCDial-S1 — Runbook (copy-paste commands)

> **One file for the full pipeline:** data → bench → zero-shot → fine-tune → eval → MultiSenNA  
> **Workspace:** `e:\MTP\earth2\`  
> **Code root:** `LULCDial-s1\`

---

## Where to run what


| Machine             | Use for                                                       |
| ------------------- | ------------------------------------------------------------- |
| **Computer system** | Extract raw `s1.tgz`, build shards + bench (disk + RAM heavy) |
| **PARAM GPU**       | Zero-shot inference, fine-tune, final eval (GPU heavy)        |
| **Laptop**          | Code, docs, smoke tests (`--max-patches`), git, monitoring    |


**Rule:** After shards + bench are built, copy only prepared artifacts to GPU server — not full 110 GB raw S1 unless needed.

---



## Key paths (always the same)

```text
e:\MTP\earth2\
├── RUNBOOK.md                          ← this file
├── Stage1_Summer_Intern_Guide.md       ← stage goals + glossary
├── log.md                              ← change history
└── LULCDial-s1\
    ├── baresoil\                       ← data-prep + eval scripts
    │   ├── build_instruct_s1.py        ← shards (train/val)
    │   ├── build_bench.py              ← MultiSenGE val bench
    │   ├── eval_zero_shot.py           ← score ZS predictions
    │   └── multisenna\
    │       └── build_bench_multisenna.py
    ├── data\baresoil_s1\
    │   ├── ai4lcc\multisenge\labels\   ← 8,157 JSON (done)
    │   ├── ai4lcc\multisenge\s1\       ← extract s1.tgz here
    │   ├── ai4lcc\multisenna\labels\   ← MultiSenNA labels (later)
    │   ├── ai4lcc\multisenna\s1\       ← MultiSenNA S1 (later)
    │   ├── shards\                     ← HF train/val shards (generated)
    │   ├── bench\v0.1\                 ← eval JSONL (generated)
    │   └── metrics\                    ← baseline + final scores
    ├── src\earthdial\train\finetune.py
    └── src\shell\data\Stage4_BareSoil_S1.json
```

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

**Model:** `akshaydudhane/EarthDial_4B_MS` (no AI4LCC training)  
**Eval data:** same bench JSONL from 1A

### 1B.1 Export questions only (no answers to model)

```powershell
cd e:\MTP\earth2\LULCDial-s1

python -m baresoil.eval_zero_shot ^
  --bench-jsonl data/baresoil_s1/bench/v0.1/ai4lcc_val.jsonl ^
  --out-metrics data/baresoil_s1/metrics/_tmp.json ^
  --dump-requests-jsonl data/baresoil_s1/bench/v0.1/ai4lcc_val_requests.jsonl
```



### 1B.2 Run EarthDial inference (GPU server)

> Inference runner script is not in `baresoil/` yet.  
> Use `ai4lcc_val_requests.jsonl` + `s1_path` per row.  
> Model gets **image + question only** (not ground-truth answers).

Save predictions as JSONL with these fields per row:

- `patch_id`
- `pred_classify`
- `pred_dialogue_turn1`
- `pred_dialogue_turn2`

Example output file:

`data/baresoil_s1/bench/v0.1/ai4lcc_val_predictions.jsonl`

### 1B.3 Score zero-shot

```powershell
cd e:\MTP\earth2\LULCDial-s1

python -m baresoil.eval_zero_shot ^
  --bench-jsonl data/baresoil_s1/bench/v0.1/ai4lcc_val.jsonl ^
  --pred-jsonl data/baresoil_s1/bench/v0.1/ai4lcc_val_predictions.jsonl ^
  --out-metrics data/baresoil_s1/metrics/earthdial_zs_baseline.json
```

**Done when:** `earthdial_zs_baseline.json` exists with F1 + dialogue accuracies.

---



## Step 1C — Fine-tune LULCDial-S1 v0.1 (GPU server)

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
| **1A** | `build_instruct_s1`, `build_bench`        | shards + `ai4lcc_val.jsonl`     |
| **1B** | `eval_zero_shot` + inference              | `earthdial_zs_baseline.json`    |
| **1C** | `finetune.py` + `Stage4_BareSoil_S1.json` | `checkpoints/LULCDial_S1_v0.1/` |
| **1D** | `eval_zero_shot` + inference              | `lulcdial_v0.1.json`            |
| **2**  | `build_bench_multisenna`                  | `multisenna_bench.jsonl`        |


---

## PARAM demo startup (optional, server only)

Use the updated demo launcher with env overrides:

```bash
cd ~/MTP/MTP/LULCDial-s1/demo
export EARTHDIAL_GPU=1
export EARTHDIAL_MODEL_PATH=/home/rihak_iitp/EarthDial_Models/EarthDial_4B_RGB
export EARTHDIAL_CONTROLLER_URL=http://0.0.0.0:40000
bash earthdial_demo.sh
```

`demo/app.py` also reads `EARTHDIAL_CONTROLLER_URL` / `EARTHDIAL_SD_WORKER_URL` if you start Streamlit manually.

---

## Related docs (deeper reading)


| File                                                                                                                 | Purpose                            |
| -------------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| `[Stage1_Summer_Intern_Guide.md](Stage1_Summer_Intern_Guide.md)`                                                     | Stage goals, glossary, timeline    |
| `[LULCDial-s1/baresoil/README.md](LULCDial-s1/baresoil/README.md)`                                                   | Data-prep commands + folder layout |
| `[BenchmarkGuide/AI4LCC/BareSoil_AI4LCC_Workflow_Guide.md](BenchmarkGuide/AI4LCC/BareSoil_AI4LCC_Workflow_Guide.md)` | PI-level workflow explanation      |
| `[AI4LCC_S1_VLM_MTech_3Stage_Roadmap.md](AI4LCC_S1_VLM_MTech_3Stage_Roadmap.md)`                                     | Full thesis roadmap (3 stages)     |
| `[log.md](log.md)`                                                                                                   | What changed in repo and when      |


