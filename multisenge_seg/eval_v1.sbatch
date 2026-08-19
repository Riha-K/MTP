#!/bin/bash
#SBATCH -J msge_v1b
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 02:00:00
#SBATCH -o /home/rihak_iitp/MTP/earth2/multisenge_seg/artifacts/slurm-%j.out
#SBATCH -e /home/rihak_iitp/MTP/earth2/multisenge_seg/artifacts/slurm-%j.err

# Test-eval run_c6_v1 best.pt (val wF1 peak, epoch 17).
set -eo pipefail
cd /home/rihak_iitp/MTP/earth2
echo "eval v1 best.pt start $(date) host=$(hostname)"

set +u
module purge
module load MLDL/Pytorch-gpu
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export PYTHONUNBUFFERED=1

python -u -m multisenge_seg.train \
  --index multisenge_seg/artifacts/patch_index.json \
  --eval-ckpt multisenge_seg/checkpoints/run_c6_v1/best.pt \
  --eval-split test \
  --batch-size 2 \
  --workers 2 \
  --out-dir multisenge_seg/checkpoints/run_c6_v1
