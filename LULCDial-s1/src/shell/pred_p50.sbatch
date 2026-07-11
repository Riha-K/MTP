#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 02:00:00
#SBATCH -J pred_p50
#SBATCH -o /home/rihak_iitp/pred_p50_%j.out
#SBATCH -e /home/rihak_iitp/pred_p50_%j.err
#
# After 1C-b finishes: predict 801 with LULCDial_S1_p50.
# sbatch ~/MTP/earth2/LULCDial-s1/src/shell/pred_p50.sbatch

module purge
module load MLDL/Pytorch-gpu
cd /home/rihak_iitp/MTP/earth2/LULCDial-s1
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
export CUDA_VISIBLE_DEVICES=0

mkdir -p data/baresoil_s1/bench/v0.1/preds/lulcdial_p50

python -m baresoil.predict_zero_shot \
  --bench-jsonl data/baresoil_s1/bench/v0.1/ai4lcc_val.jsonl \
  --s1-root data/baresoil_s1/ai4lcc/multisenge/s1_val_bench \
  --checkpoint /home/rihak_iitp/MTP/earth2/LULCDial-s1/checkpoints/LULCDial_S1_p50 \
  --out-pred-jsonl data/baresoil_s1/bench/v0.1/preds/lulcdial_p50/ai4lcc_val_predictions.jsonl \
  --resume
