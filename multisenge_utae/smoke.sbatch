#!/bin/bash
#SBATCH -J utae_smoke
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 00:30:00
#SBATCH -o /home/rihak_iitp/MTP/earth2/multisenge_utae/artifacts/slurm-smoke-%j.out
#SBATCH -e /home/rihak_iitp/MTP/earth2/multisenge_utae/artifacts/slurm-smoke-%j.err

set -euo pipefail
cd /home/rihak_iitp/MTP/earth2
mkdir -p multisenge_utae/artifacts multisenge_utae/checkpoints

module purge
module load MLDL/Pytorch-gpu
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

python -m multisenge_utae.train \
  --index multisenge_seg/artifacts/patch_index.json \
  --num-classes 6 --mode head --epochs 2 \
  --batch-size 1 --workers 0 \
  --max-train 4 --max-val 2 \
  --full-class-weights \
  --out-dir multisenge_utae/checkpoints/run_c6_head_smoke
