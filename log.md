# LULCDial-S1 / LULCDial — Change Log

Running record of code, data-pipeline, and config changes for this thesis workspace.

**How to use:** When you (or the agent) make a meaningful change, add a new dated section **at the top** of [Entries](#entries) (newest first). Keep each entry short: what changed, why, which files/paths, and any commands or outputs worth remembering.

**Note:** Do not run full shard/bench builds on a laptop without enough RAM/disk. Use `--max-patches` for smoke tests; run full builds on professor PC when ready.

---

## Entries

### 2026-07-10 — Stage 1B zero-shot inference runner

**Why:** Start Stage 1B (EarthDial_4B_MS baseline before fine-tune). Scorer already existed; model batch runner did not. PARAM has shards/bench but not full raw S1.

**Added:**
- `baresoil/predict_zero_shot.py` — S1 VH → `s1` norm → `sequential_vit_features` → classify + 2-turn dialogue preds
- `baresoil/pack_bench_s1.py` — copy only bench-referenced TIFFs (~801) for upload
- `RUNBOOK.md` / `baresoil/README.md` — Stage 1B pack → infer → score commands

**Next on user side:** pack `s1_val_bench`, scp to PARAM, confirm `EarthDial_4B_MS` weights, smoke `--max-samples 20`, then full + `eval_zero_shot`.

---

### 2026-07-10 — Stage 1A full build + MultiSenNA bench; artifacts on PARAM

**Why:** Finish MultiSenGE train/val shards + GE/NA benches on remote CPU, then stage prepared artifacts on PARAM for Stage 1B/1C (no raw 110 GB S1 on GPU login).

**Code:**
- `baresoil/build_instruct_s1.py` — stream via `Dataset.from_generator` (fixes `MemoryError` on full train save)
- `baresoil/taxonomy.py` — add MultiSenNA class **15 = Beaches, Sand**; classify options include 1–15
- Commit `fdd0960` pushed to `origin/main`

**Remote CPU (`D:\Riha\earth2`) outputs:**
- Train shard: `14710` samples / `7355` patches (`bad_tif: 1`)
- Val shard: `1602` samples / `801` patches
- GE bench: `bench/v0.1/ai4lcc_val.jsonl` → **801** rows
- MultiSenNA bench: `12115` rows (`skipped_missing_s1: 143`)

**PARAM (`~/MTP/earth2`):**
- `git pull` → `fdd0960`
- Copied: `shards/ai4lcc_ge_train_{train,val}/`, `bench/multisenna/v0.1/multisenna_bench.jsonl` (+ summary)
- Verified: GE `801` lines; MultiSenNA `12115` lines; both shard dirs present

**Not done yet:** Stage 1B EarthDial inference runner; Stage 1C fine-tune path update + launch.

---

### 2026-07-09 — PARAM demo config defaults from old EarthDial copy

**Why:** Port only server-specific demo startup settings from downloaded `EarthDial-old`, without touching LULCDial training/eval pipeline.

**Changed:**
- `LULCDial-s1/demo/app.py` — controller/SD worker defaults now from env vars (`EARTHDIAL_CONTROLLER_URL`, `EARTHDIAL_SD_WORKER_URL`), PARAM fallback `http://login01:40000`
- `LULCDial-s1/demo/earthdial_demo.sh` — env-based GPU/model path defaults (`EARTHDIAL_GPU`, `EARTHDIAL_MODEL_PATH`)
- `RUNBOOK.md` — added PARAM demo startup block

**Not changed:** `baresoil/*`, `finetune.py`, `Stage4_BareSoil_S1.json`, training/eval logic.

---

### 2026-07-09 — Add single RUNBOOK.md command file

**Why:** User needed one main file with copy-paste commands for full pipeline (data → ZS → train → eval → MultiSenNA).

**Changed:**
- Added `RUNBOOK.md` at repo root with stage checklist and all command blocks
- Updated root `README.md` to point to `RUNBOOK.md` as primary command reference

---

### 2026-07-08 — Add MultiSenNA folders + bench builder

**Why:** Prepare Stage 2 transfer-eval data layout now, while short-time server access is available.

**Changed:**
- Added folder placeholders:
  - `LULCDial-s1/data/baresoil_s1/ai4lcc/multisenna/labels/.gitkeep`
  - `LULCDial-s1/data/baresoil_s1/ai4lcc/multisenna/s1/.gitkeep`
  - `LULCDial-s1/data/baresoil_s1/bench/multisenna/v0.1/.gitkeep`
- Added `LULCDial-s1/baresoil/multisenna/build_bench_multisenna.py`
  - Builds MultiSenNA classify + 2-turn dialogue bench JSONL
  - Supports `--max-samples` smoke run
  - Writes summary JSON with counts and skipped missing-S1 rows
- Added `LULCDial-s1/baresoil/multisenna/__init__.py`
- Updated `LULCDial-s1/baresoil/README.md` with MultiSenNA storage paths and commands

**Not changed:** fine-tune config and Stage 4 training code.

---

### 2026-07-08 — Add Stage 1B zero-shot eval scorer (no fine-tune changes)

**Why:** Prepare zero-shot baseline workflow now, while waiting for the allocated system, without touching Stage 4 fine-tune path.

**Changed:**
- Added `LULCDial-s1/baresoil/eval_zero_shot.py`
  - Exports bench request rows (`--dump-requests-jsonl`)
  - Scores predictions JSONL against bench ground truth (`--pred-jsonl`)
  - Writes metrics JSON (`classification.example_f1`, dialogue turn-1/turn-2 set-match accuracy)
- Updated `LULCDial-s1/baresoil/README.md` with Stage 1B command flow (request export + scoring)

**Not changed:** `finetune.py`, `src/shell/data/Stage4_BareSoil_S1.json`, train shard pipeline.

---

### 2026-07-08 — Sync docs to earth2 workspace; drop caption from plan

**Why:** Local folder is `e:\MTP\earth2\` (not `LULCDial\`). Caption task removed from code; roadmap and guides still mentioned it. `readGuide.md` duplicated `README.md`.

**Changed:**
- `Stage4_BareSoil_S1.json` — paths → `e:/MTP/earth2/LULCDial-s1/data/...`
- `AI4LCC_S1_VLM_MTech_3Stage_Roadmap.md` — caption removed; 2 tasks/patch (~16k QA); workspace `earth2`
- `Stage1_Summer_Intern_Guide.md`, `BareSoil_AI4LCC_Workflow_Guide.md`, `baresoil/README.md` — same
- Deleted `readGuide.md` (use root `README.md`)

**Naming:** Workspace folder = `earth2`; thesis product = **LULCDial-S1**; code repo = `LULCDial-s1/`.

---

### 2026-07-07 — Inline study comments in baresoil core modules

**Why:** Add readable inline notes while learning the pipeline (`taxonomy` → `patch_meta` → `s1_vh_io`). No logic changes.

**Changed:**
- `LULCDial-s1/baresoil/taxonomy.py` — comment tweak in `ai4lcc_names_from_ids`
- `LULCDial-s1/baresoil/patch_meta.py` — notes on JSON parsing, `iter_patches`, `pick_s1_path`
- `LULCDial-s1/baresoil/s1_vh_io.py` — notes on rasterio, VH dB conversion, PIL float shard, preview PNG

---

### 2026-07-07 — Prune BenchmarkGuide; remove Bench2.0 duplicates

**Why:** Project only needs AI4LCC guides under `BenchmarkGuide/`. BigEarthNet, DynamicWorld, OpenEarthMap, and old `BareSoil_S1_VLM_Dataset_Guide.md` (7-class plan) are out of scope. `Bench2.0/AI4LCC/` duplicated `BenchmarkGuide/AI4LCC/` with stale `earth2` / `EarthDial-main` paths.

**Kept (correct / latest):**

- `BenchmarkGuide/AI4LCC/BareSoil_AI4LCC_Workflow_Guide.md` — LULCDial paths, current pipeline
- `BenchmarkGuide/AI4LCC/MultiSenGE_AI4LCC_Complete_Analysis.md` — LULCDial-s1 module names
- `BenchmarkGuide/AI4LCC/multiSenge_AI4LCC.pdf` — moved from `Bench2.0/AI4LCC/` (only copy)
- `Stage1_Summer_Intern_Guide.md` (repo root) — 2 QA/patch, `patch_meta.py` / `s1_vh_io.py` names

**Removed:**

- `Bench2.0/` (entire folder)
- `BenchmarkGuide/BigEarthnet/`, `DynamiWorld/`, `OpenEarthMap/`
- `BenchmarkGuide/BareSoil_S1_VLM_Dataset_Guide.md`

---

### 2026-07-07 — Connect repo to GitHub; add LULCDial-s1 + updated docs

**Why:** Sync local `e:\MTP\LULCDial\` with [Riha-K/MTP](https://github.com/Riha-K/MTP.git). Update already-pushed docs; add new code, guides, and label JSONs. Exclude ~115 GB S1 imagery via `.gitignore`.

**Changed:**

- Added `.gitignore` (excludes `s1/*.tif`, shards, previews, checkpoints, `*.tgz`)
- Updated: roadmap, analysis, workflow guides, `README.md` (from `readGuide.md`)
- Added: `LULCDial-s1/` (EarthDial fork + `baresoil/` pipeline), `log.md`, `Stage1_Summer_Intern_Guide.md`, label JSONs, papers

**Not pushed:** S1 GeoTIFFs, HF shards, bench previews, checkpoints (local only).

**Push fix:** Redacted hardcoded Hugging Face token from `LULCDial-s1/Dockerfile` and `LULCDial-s1/src/EarthDial_AppWrapper_old.yaml` (GitHub push protection blocked commit).

---

### 2026-07-07 — Cleared stale bench artifacts before full rebuild

**Why:** Old `ai4lcc_val.jsonl` used 3-task schema (caption fields, old prompts). Preview PNGs were from a partial debug run. User will rerun full `build_bench.py` after shard build.

**Changed:**

- Emptied `LULCDial-s1/data/baresoil_s1/bench/v0.1/ai4lcc_val.jsonl`
- Removed all PNGs from `LULCDial-s1/data/baresoil_s1/bench/previews/` (folder kept)

---

### 2026-07-07 — Rename workspace paths: earth2 → LULCDial, EarthDial-main → LULCDial-s1

**Why:** Docs and configs still pointed at old folder names (`e:\MTP\earth2\`, `EarthDial-main/`). Actual workspace is `e:\MTP\LULCDial\` with code in `LULCDial-s1/`.

**Changed:**

- `AI4LCC_S1_VLM_MTech_3Stage_Roadmap.md`, `readGuide.md`, `Stage1_Summer_Intern_Guide.md`, `EarthDial_Complete_Analysis.md`
- `BenchmarkGuide/AI4LCC/BareSoil_AI4LCC_Workflow_Guide.md`, `BenchmarkGuide/AI4LCC/MultiSenGE_AI4LCC_Complete_Analysis.md`
- `LULCDial-s1/baresoil/README.md`, `build_instruct_s1.py`, `sample_qa.txt`, `requirements.txt`
- `LULCDial-s1/src/shell/data/Stage4_BareSoil_S1.json` — shard paths now `e:/MTP/LULCDial/LULCDial-s1/data/...`
- `log.md` — title + historical path references updated

---

### 2026-07-07 — Rename baresoil modules for clarity

**Why:** `ai4lcc.py` and `s1_io.py` names were unclear; renamed to match what each file does.

**Changed:**

- `baresoil/ai4lcc.py` → `baresoil/patch_meta.py` (patch metadata + label JSON + S1 path helpers)
- `baresoil/s1_io.py` → `baresoil/s1_vh_io.py` (VH GeoTIFF read + PIL/PNG conversion)
- Updated imports in `build_instruct_s1.py`, `build_bench.py`, `README.md`, guides

---

### 2026-07-07 — Drop caption; classify + 2-turn dialogue only

**Why:** Caption duplicated classify (verbatim class list). Chose Option D1 from `sample_qa.txt`: keep structured classify + 2-turn dialogue (natural/ag subset).

**Changed:**

- `LULCDial-s1/baresoil/instruct_templates.py` — removed `build_caption_qa()` and `[caption]` token
- `LULCDial-s1/baresoil/build_instruct_s1.py` — 2 QA rows per patch (classify + dialogue)
- `LULCDial-s1/baresoil/build_bench.py` — removed caption_question/caption_answer fields
- `LULCDial-s1/baresoil/sample_qa.txt` — marked D1 as active in code
- `LULCDial-s1/baresoil/README.md` — ~14.7k QA count (8157 × 2)

**Scale:** 8,157 patches × 2 tasks ≈ 16,314 max instruction rows (before split).

---

### 2026-07-03 — Simpler EarthDial-style instruction templates

**Why:** Prompts were too technical ("official OCSGE", "Sentinel-1 VH backscatter"). Aligned with EarthDial AID-style short questions while keeping 14-class option list for classify.

**Changed:** `LULCDial-s1/baresoil/instruct_templates.py` — shorter human questions; caption answer uses **verbatim** class names (pending professor choice vs natural sentence).

**Review doc for PI:** `LULCDial-s1/baresoil/sample_qa.txt` (shows classify, caption A/B, dialogue samples).

---

### 2026-07-02 (restore) — Re-applied 14-class OCSGE + S1 date fix after undo

**Context:** User undid prior session changes (including accidental rebuild runs). Restored code and this log only — **no build commands run**.

**Code restored** (`LULCDial-s1/baresoil/`):


| File                    | Purpose / change                                                                         |
| ----------------------- | ---------------------------------------------------------------------------------------- |
| `taxonomy.py`           | Official **14 OCSGE** class map and helpers (no 7-class unified remapping).              |
| `instruct_templates.py` | Classify + caption + 2-turn dialogue QA using OCSGE names.                               |
| `ai4lcc.py`             | `PatchMeta` with `label_names`, `dominant_class_name`; added `pick_available_s1_file()`. |
| `build_instruct_s1.py`  | 3 QA per patch; uses `pick_available_s1_file` for full patch coverage.                   |
| `build_bench.py`        | Val JSONL with classify/caption/dialogue; ASCII print; uses `pick_available_s1_file`.    |


**S1 date fix (why):** JSON lists many dates per patch; old code picked median of full list. Often that exact file was not on disk. New code picks median among **files that exist** — verified 8,157/8,157 patches resolvable without re-download.

**Expected full outputs (when you run builds later):**


| Artifact     | Patches | QA / rows |
| ------------ | ------- | --------- |
| Train shards | 7,356   | 22,068 QA |
| Val shards   | 801     | 2,403 QA  |
| Bench JSONL  | —       | 801 rows  |


**Risk:** Full shard build loads all images into RAM before save — can hit `MemoryError` on laptop (~11 min then crash observed). Use professor PC or `--max-patches 100` on laptop.

---

### 2026-07-02 — 14-class OCSGE templates, deps, partial shards & bench

**Goal:** Switch from 7-class bare-soil remapping to official **14 OCSGE** names; install deps; first shard/bench attempt.

**Dependencies:** `pip install -r LULCDial-s1/baresoil/requirements.txt`

**First build (before S1 fix):** Only 1,666 patches matched (train 1,510, val 156) because median JSON date often missing on disk.

**Partial outputs on disk (may be stale after undo/rebuild attempts):**

- `data/baresoil_s1/shards/ai4lcc_ge_train_train/` — partial
- `data/baresoil_s1/shards/ai4lcc_ge_train_val/` — partial
- `data/baresoil_s1/bench/v0.1/ai4lcc_val.jsonl` — may contain 801 rows from one successful bench run

**Next when user requests:** Re-run shard build on suitable machine with restored code.

---

