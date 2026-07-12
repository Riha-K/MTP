#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 16:00:00
#SBATCH -J pred_na
#SBATCH -o /home/rihak_iitp/pred_na_%j.out
#SBATCH -e /home/rihak_iitp/pred_na_%j.err
#
# MultiSenNA full transfer predict (~12k) with LULCDial_S1_v0.1.
# ETA ~8–12 h (scale from ~45 min / 801 GE). Run smoke first.
# sbatch ~/MTP/earth2/LULCDial-s1/src/shell/pred_multisenna_v0.1.sbatch

module purge
module load MLDL/Pytorch-gpu
cd /home/rihak_iitp/MTP/earth2/LULCDial-s1
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
export CUDA_VISIBLE_DEVICES=0

S1_ROOT="data/baresoil_s1/ai4lcc/multisenna/s1_na_bench"
if [ ! -d "$S1_ROOT" ]; then
  S1_ROOT="data/baresoil_s1/ai4lcc/multisenna/s1"
fi

BENCH="data/baresoil_s1/bench/multisenna/v0.1/multisenna_bench.jsonl"
if [ ! -f "$BENCH" ]; then
  BENCH="data/baresoil_s1/bench/v0.1/multisenna_bench.jsonl"
fi

mkdir -p data/baresoil_s1/bench/v0.1/preds/lulcdial_v0.1_multisenna

python -m baresoil.predict_zero_shot \
  --bench-jsonl "$BENCH" \
  --s1-root "$S1_ROOT" \
  --checkpoint /home/rihak_iitp/MTP/earth2/LULCDial-s1/checkpoints/LULCDial_S1_v0.1 \
  --out-pred-jsonl data/baresoil_s1/bench/v0.1/preds/lulcdial_v0.1_multisenna/predictions.jsonl \
  --resume
