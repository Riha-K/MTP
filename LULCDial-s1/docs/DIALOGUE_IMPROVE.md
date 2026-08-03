# Dialogue improvement plan (2-turn T1 / T2)

> Goal: raise MultiSenGE dialogue scores without changing the primary classify story.  
> Related code: `lulcdial/eval_zero_shot.py`, `lulcdial/instruct_templates.py`  
> Roadmap: see § below + `ROADMAP.md`

---

## Machine guide (what needs which computer)

| Step | Where | Heavy? | Why |
|------|-------|--------|-----|
| **A. Soft metrics (dialogue example F1)** | **Laptop only** | No | Re-score existing preds; no GPU, no full S1 |
| **B. Stronger dialogue prompts in templates** | Laptop (code) | No | Edit only |
| **C. Rebuild train/val shards + GE bench** | **Sir PC** | Yes (disk / many TIFFs) | Images baked into shards; need full `multisenge/s1` |
| **D. Fine-tune / continue-train + re-predict** | **PARAM GPU** | Yes (GPU) | Same as earlier FT / pred jobs |
| **E. Update tables / writeup** | Laptop | No | Docs only |

**You do not need sir PC for Step A.** Do A now on the laptop.  
Sir PC + PARAM are only needed when you decide to retrain with the new dialogue prompts (Steps C–D).

---

## Step A — Soft dialogue metrics (laptop, now)

Exact set-match understates dialogue quality (classify F1≈0.81 but T1 set-match≈0.13). The scorer now also reports:

- `dialogue.turn1_example_f1`
- `dialogue.turn2_example_f1`

Same F1 definition as classify, applied to each dialogue turn.

```bat
cd /d e:\MTP\earth2\LULCDial-s1
set PYTHONPATH=%CD%;%CD%\src

python -m lulcdial.eval_zero_shot ^
  --bench-jsonl data/lulcdial_s1/bench/v0.1/ai4lcc_test.jsonl ^
  --pred-jsonl data/lulcdial_s1/bench/v0.1/preds/lulcdial_v0.1/ai4lcc_test_predictions.jsonl ^
  --out-metrics data/lulcdial_s1/metrics/v0.1/lulcdial_v0.1.json

python -m lulcdial.eval_zero_shot ^
  --bench-jsonl data/lulcdial_s1/bench/multisenna/v0.1/multisenna_bench.jsonl ^
  --pred-jsonl data/lulcdial_s1/bench/v0.1/preds/lulcdial_v0.1_multisenna/predictions.jsonl ^
  --out-metrics data/lulcdial_s1/metrics/v0.1/lulcdial_v0.1_multisenna.json
```

If MultiSenNA bench is missing locally, run only the GE command (or copy the JSONL from PARAM).

**Paper rule:** keep reporting set-match as the strict metric; also report dialogue example F1 as secondary.

---

## Step B — Format-aligned dialogue prompts (code done)

`build_dialogue_turns` now:

- Lists the OCSGE option set (like classify)
- Asks for **comma-separated class names only** (no sentences)
- Keeps turn-1 GT identical to classify answer (consistency)

This is already in `lulcdial/instruct_templates.py`. **Old shards/bench still use open prompts** until you rebuild (Step C).

---

## Step C — Rebuild shards + bench (sir PC)

Only when ready to retrain:

```bat
cd /d D:\Riha\earth2\LULCDial-s1
set PYTHONPATH=%CD%;%CD%\src
git pull

python -m lulcdial.build_instruct_s1 --labels-dir data/lulcdial_s1/ai4lcc/multisenge/labels --s1-dir data/lulcdial_s1/ai4lcc/multisenge/s1 --out-dir data/lulcdial_s1/shards/ai4lcc_ge_train --split all --train-ratio 0.7

python -m lulcdial.build_bench --labels-dir data/lulcdial_s1/ai4lcc/multisenge/labels --s1-dir data/lulcdial_s1/ai4lcc/multisenge/s1 --out-jsonl data/lulcdial_s1/bench/v0.1/ai4lcc_test.jsonl --train-ratio 0.7
```

Optional dialogue-heavy continue-train later: duplicate dialogue samples in a small script (documented when needed).

---

## Step D — PARAM retrain + predict

1. Upload new shards (and new `ai4lcc_test.jsonl` if questions changed).  
2. `sbatch` train → new or continued checkpoint (name e.g. `LULCDial_S1_v0.1_dlg`).  
3. Predict GE (+ MultiSenNA if desired).  
4. Score with Step A commands.  
5. **Never** co-schedule two of your jobs on the same node.

---

## Step E — Paper / package

Update Table 1 footnote + `sar_lc_bench_v0.1/EVAL_PROTOCOL.md` with dual dialogue metrics after Step A numbers exist; update again after Step D.

---

## Status

| Step | Status |
|------|--------|
| A scorer + dialogue F1 | ✅ done on laptop for GE FT + ZS |
| B templates | ✅ code ready — needs C+D to affect model |
| C rebuild | ⬜ sir PC |
| D FT + predict | ⬜ PARAM |
| E docs with new numbers | ✅ soft metrics in tables; model FT pending |

### Step A results (GE, existing LULCDial preds)

| Metric | Value |
|--------|-------|
| classify example F1 | **0.812** |
| T1 set-match | 0.134 |
| T2 set-match | 0.390 |
| **T1 example F1** | **0.813** (≈ classify) |
| **T2 example F1** | **0.870** |

Interpretation: dialogue content is already strong; exact set-match is the bottleneck. Format-aligned FT (C+D) aims to lift set-match without hurting classify F1.
