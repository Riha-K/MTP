#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 02:00:00
#SBATCH -J pred_na_sm
#SBATCH -o /home/rihak_iitp/pred_na_smoke_%j.out
#SBATCH -e /home/rihak_iitp/pred_na_smoke_%j.err
#
# MultiSenNA smoke (100 samples) with LULCDial_S1_v0.1 (70/30 GE FT).
# sbatch ~/MTP/earth2/LULCDial-s1/src/shell/pred_multisenna_smoke.sbatch

module purge
module load MLDL/Pytorch-gpu
cd /home/rihak_iitp/MTP/earth2/LULCDial-s1
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
export CUDA_VISIBLE_DEVICES=0

# Prefer packed NA bench only if it actually contains TIFFs; else full multisenna/s1 tree.
S1_ROOT="data/lulcdial_s1/ai4lcc/multisenna/s1"
NA_PACK="data/lulcdial_s1/ai4lcc/multisenna/s1_na_bench"
if [ -d "$NA_PACK" ] && [ -n "$(ls -A "$NA_PACK" 2>/dev/null | head -1)" ]; then
  S1_ROOT="$NA_PACK"
fi
echo "=== S1_ROOT=$S1_ROOT ==="

BENCH="data/lulcdial_s1/bench/multisenna/v0.1/multisenna_bench.jsonl"
if [ ! -f "$BENCH" ]; then
  BENCH="data/lulcdial_s1/bench/v0.1/multisenna_bench.jsonl"
fi

mkdir -p data/lulcdial_s1/bench/v0.1/preds/lulcdial_v0.1_multisenna_smoke

python -m lulcdial.predict_zero_shot \
  --bench-jsonl "$BENCH" \
  --s1-root "$S1_ROOT" \
  --checkpoint /home/rihak_iitp/MTP/earth2/LULCDial-s1/checkpoints/LULCDial_S1_v0.1 \
  --out-pred-jsonl data/lulcdial_s1/bench/v0.1/preds/lulcdial_v0.1_multisenna_smoke/predictions.jsonl \
  --max-samples 100 \
  --resume
