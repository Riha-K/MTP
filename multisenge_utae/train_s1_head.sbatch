#!/bin/bash
# Task M ablation: U-TAE S1-only (2ch VV/VH) 6c P4 head — like paper ConvLSTM-S1.
#   sbatch --exclude=ragpu004,ragpu005,ragpu007 multisenge_utae/train_s1_head.sbatch
#SBATCH -J utae_s1_head
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH -o /home/rihak_iitp/MTP/earth2/multisenge_utae/artifacts/slurm-s1-head-%j.out
#SBATCH -e /home/rihak_iitp/MTP/earth2/multisenge_utae/artifacts/slurm-s1-head-%j.err

set -eo pipefail
cd /home/rihak_iitp/MTP/earth2
mkdir -p multisenge_utae/artifacts multisenge_utae/checkpoints

module purge
module load MLDL/Pytorch-gpu
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

python -m multisenge_utae.train \
  --index multisenge_seg/artifacts/patch_index.json \
  --data-root LULCDial-s1/data/lulcdial_s1/ai4lcc/multisenge \
  --num-classes 6 \
  --mode head \
  --modality s1 \
  --epochs 80 \
  --batch-size 2 \
  --workers 2 \
  --out-dir multisenge_utae/checkpoints/run_c6_s1_head_v0

python -m multisenge_utae.plot_history \
  multisenge_utae/checkpoints/run_c6_s1_head_v0/history.json \
  --title "U-TAE S1-only P4 head (6-class)" \
  || echo "plot skipped"
