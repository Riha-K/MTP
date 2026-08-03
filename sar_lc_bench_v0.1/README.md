# SAR-LC-Bench v0.1

**Sentinel-1 VH · OCSGE 14-class · multi-label classify + 2-turn dialogue**

SAR-LC-Bench evaluates patch-level land-cover vision–language models on the official **14-class OCSGE** taxonomy used by AI4LCC MultiSenGE (Grand Est, France). It is **not** a new satellite archive: the contribution is a frozen eval protocol, instruction templates, metrics, and a public leaderboard.

| Item | Value |
|------|-------|
| Version | **v0.1** |
| Split | Frozen **70/30** (MD5 of patch stem) |
| In-domain test | MultiSenGE held-out **2497** patches |
| Transfer (optional row) | MultiSenNA **11939** patches (no NA training) |
| Primary metric | **example F1** (multi-label set F1 per patch, mean) |
| Secondary | Dialogue turn-1 / turn-2 **set-match accuracy** |

Companion model (reported baseline): **LULCDial-S1** (EarthDial_4B_MS fine-tuned on MultiSenGE train).

---

## What is in this package

| File | Purpose |
|------|---------|
| [`EVAL_PROTOCOL.md`](EVAL_PROTOCOL.md) | Tasks, taxonomy, split rule, scoring |
| [`BENCH_MANIFEST_v0.1.json`](BENCH_MANIFEST_v0.1.json) | Paths, counts, checksums |
| [`leaderboard.csv`](leaderboard.csv) | Official reported scores |
| [`data/README.md`](data/README.md) | How to obtain labels / S1 rasters |
| [`PUBLISH.md`](PUBLISH.md) | How to release this as a **separate public GitHub repo** |

Machine-readable bench rows live in the research tree as  
`ai4lcc_test.jsonl` (and MultiSenNA `multisenna_bench.jsonl`) — see manifest.  
**Do not** commit full MultiSenGE/MultiSenNA TIFF archives into Git.

---

## Quick start (score predictions)

1. Obtain bench JSONL + S1 TIFFs for the listed `patch_id`s (see `data/README.md`).
2. Produce a predictions JSONL keyed by `patch_id` with:
   - `pred_classify`
   - `pred_dialogue_turn1`
   - `pred_dialogue_turn2`
3. Score with the reference evaluator (from the LULCDial-S1 codebase):

```bash
python -m lulcdial.eval_zero_shot \
  --bench-jsonl path/to/ai4lcc_test.jsonl \
  --pred-jsonl path/to/your_predictions.jsonl \
  --out-metrics path/to/metrics.json
```

Report **`classification.example_f1`** as the primary leaderboard score.

---

## Citation

Paper / thesis citation TBA. Until then, cite AI4LCC MultiSenGE for the underlying rasters and labels, and cite this bench protocol when using the eval package.

---

## License

- **Code & protocol text** in this folder: MIT (see `LICENSE`).
- **Imagery & labels:** follow AI4LCC / MultiSenGE / MultiSenNA license terms (not redistributed here by default).
