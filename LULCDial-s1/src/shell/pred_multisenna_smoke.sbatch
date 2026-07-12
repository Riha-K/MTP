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
# MultiSenNA smoke (100 samples) with LULCDial_S1_v0.1 — verify S1 paths before full ~12k.
# sbatch ~/MTP/earth2/LULCDial-s1/src/shell/pred_multisenna_smoke.sbatch

module purge
module load MLDL/Pytorch-gpu
cd /home/rihak_iitp/MTP/earth2/LULCDial-s1
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
export CUDA_VISIBLE_DEVICES=0

# Prefer packed TIFF folder if present; else full multisenna/s1
S1_ROOT="data/baresoil_s1/ai4lcc/multisenna/s1_na_bench"
if [ ! -d "$S1_ROOT" ]; then
  S1_ROOT="data/baresoil_s1/ai4lcc/multisenna/s1"
fi

BENCH="data/baresoil_s1/bench/multisenna/v0.1/multisenna_bench.jsonl"
if [ ! -f "$BENCH" ]; then
  BENCH="data/baresoil_s1/bench/v0.1/multisenna_bench.jsonl"
fi

mkdir -p data/baresoil_s1/bench/v0.1/preds/lulcdial_v0.1_multisenna_smoke

python -m baresoil.predict_zero_shot \
  --bench-jsonl "$BENCH" \
  --s1-root "$S1_ROOT" \
  --checkpoint /home/rihak_iitp/MTP/earth2/LULCDial-s1/checkpoints/LULCDial_S1_v0.1 \
  --out-pred-jsonl data/baresoil_s1/bench/v0.1/preds/lulcdial_v0.1_multisenna_smoke/predictions.jsonl \
  --max-samples 100 \
  --resume
