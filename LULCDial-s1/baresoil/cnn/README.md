# ResNet multi-label S1 baseline (fair vs LULCDial)

Same **70/30 MD5 split**, same **14 OCSGE** classes, same **patch example F1** via `eval_zero_shot.py`.

## Files

| Module | Role |
|--------|------|
| `split.py` | MD5 train/val bucket (matches VLM builders) |
| `dataset.py` | S1 VH → 3ch ImageNet tensor; multi-hot labels |
| `model.py` | ResNet-18/50 + 14-logit head |
| `train.py` | BCEWithLogits train → `best.pt` / `last.pt` |
| `predict.py` | Bench → `pred_classify` JSONL |

## Smoke (any machine with torch)

```bash
cd LULCDial-s1
set PYTHONPATH=%CD%   # Windows cmd; Linux: export PYTHONPATH="$(pwd)"

python -m baresoil.cnn.train \
  --labels-dir data/baresoil_s1/ai4lcc/multisenge/labels \
  --s1-dir data/baresoil_s1/ai4lcc/multisenge/s1 \
  --out-dir checkpoints/resnet18_s1_smoke \
  --epochs 1 --batch-size 8 --max-samples 64 --num-workers 0
```

## Full GPU (PARAM)

```bash
sbatch src/shell/train_resnet_v0.2.sbatch
sbatch src/shell/pred_resnet_v0.2.sbatch   # after train finishes
```

Needs on PARAM: `labels/` + full `s1/` for train; `s1_test_bench_v0.2/` + `ai4lcc_test.jsonl` for pred.

## Score only (if you already have preds)

```bash
python -m baresoil.eval_zero_shot \
  --bench-jsonl data/baresoil_s1/bench/v0.2/ai4lcc_test.jsonl \
  --pred-jsonl data/baresoil_s1/bench/v0.2/preds/resnet18/ai4lcc_test_predictions.jsonl \
  --out-metrics data/baresoil_s1/metrics/v0.2/resnet18_s1.json
```
