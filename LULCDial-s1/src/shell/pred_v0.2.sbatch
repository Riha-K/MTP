#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 05:00:00
#SBATCH -J pred_v02
#SBATCH -o /home/rihak_iitp/pred_v02_%j.out
#SBATCH -e /home/rihak_iitp/pred_v02_%j.err
#
# Predict 70/30 test (~2497) with LULCDial_S1_v0.2 after fine-tune.
# sbatch ~/MTP/earth2/LULCDial-s1/src/shell/pred_v0.2.sbatch

module purge
module load MLDL/Pytorch-gpu
cd /home/rihak_iitp/MTP/earth2/LULCDial-s1
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
export CUDA_VISIBLE_DEVICES=0

mkdir -p data/baresoil_s1/bench/v0.2/preds/lulcdial_v0.2

python -m baresoil.predict_zero_shot \
  --bench-jsonl data/baresoil_s1/bench/v0.2/ai4lcc_test.jsonl \
  --s1-root data/baresoil_s1/ai4lcc/multisenge/s1_test_bench_v0.2 \
  --checkpoint /home/rihak_iitp/MTP/earth2/LULCDial-s1/checkpoints/LULCDial_S1_v0.2 \
  --out-pred-jsonl data/baresoil_s1/bench/v0.2/preds/lulcdial_v0.2/ai4lcc_test_predictions.jsonl \
  --resume
