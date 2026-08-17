#!/bin/bash
#SBATCH -J msge_cnn
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
# IMPORTANT: absolute paths — Slurm does NOT expand ~
#SBATCH -o /home/rihak_iitp/MTP/earth2/multisenge_seg/artifacts/slurm-%j.out
#SBATCH -e /home/rihak_iitp/MTP/earth2/multisenge_seg/artifacts/slurm-%j.err

set -euo pipefail
cd /home/rihak_iitp/MTP/earth2
mkdir -p multisenge_seg/artifacts multisenge_seg/checkpoints

module purge
module load MLDL/Pytorch-gpu
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

python -m multisenge_seg.build_index \
  --data-root LULCDial-s1/data/lulcdial_s1/ai4lcc/multisenge \
  --out multisenge_seg/artifacts/patch_index.json

python -m multisenge_seg.train \
  --index multisenge_seg/artifacts/patch_index.json \
  --data-root LULCDial-s1/data/lulcdial_s1/ai4lcc/multisenge \
  --num-classes 6 \
  --epochs 80 \
  --batch-size 2 \
  --workers 2 \
  --out-dir multisenge_seg/checkpoints/run_c6_v0
