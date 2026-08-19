#!/bin/bash
#SBATCH -J msge_c10v1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH -o /home/rihak_iitp/MTP/earth2/multisenge_seg/artifacts/slurm-%j.out
#SBATCH -e /home/rihak_iitp/MTP/earth2/multisenge_seg/artifacts/slurm-%j.err

# 10-class v1: seed 42, EarlyStop on kappa, boost grass + rare urban, downweight water (FP).
# v0 replicate stays run_c10_v0 (test W-F1 0.8711). Do not overwrite it.
# If OOM: --batch-size 4 --accum-steps 4
set -eo pipefail
cd /home/rihak_iitp/MTP/earth2
mkdir -p multisenge_seg/artifacts multisenge_seg/checkpoints
echo "train c10 v1 start $(date) host=$(hostname)"

set +u
module purge
module load MLDL/Pytorch-gpu

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export PYTHONUNBUFFERED=1
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.__version__)"

python -u -m multisenge_seg.train \
  --index multisenge_seg/artifacts/patch_index.json \
  --data-root LULCDial-s1/data/lulcdial_s1/ai4lcc/multisenge \
  --num-classes 10 \
  --epochs 80 \
  --batch-size 8 \
  --accum-steps 2 \
  --stats-patches 0 \
  --full-class-weights \
  --seed 42 \
  --monitor kappa \
  --class-boost 8:2.5,4:1.5,10:0.6 \
  --workers 2 \
  --out-dir multisenge_seg/checkpoints/run_c10_v1
