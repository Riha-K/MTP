#!/bin/bash
#SBATCH -J msge_cnn
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -t 24:00:00
#SBATCH -o multisenge_seg/artifacts/slurm-%j.out
#SBATCH -e multisenge_seg/artifacts/slurm-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/MTP/earth2}"
# module load MLDL/Pytorch-gpu   # PARAM: uncomment / adjust
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
