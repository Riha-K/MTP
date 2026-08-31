#!/bin/bash
# PARAM submit (from ~/MTP/earth2):
#   sbatch --exclude=ragpu004,ragpu005,ragpu007 multisenge_utae/train.sbatch
# Exclude: ragpu005/007 down; ragpu004 often overloaded (3 jobs on gpu:2).
# Check GPU nodes: sinfo -N -p gpu -o "%N %T %C %G"
#SBATCH -J msge_utae_head
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH -o /home/rihak_iitp/MTP/earth2/multisenge_utae/artifacts/slurm-%j.out
#SBATCH -e /home/rihak_iitp/MTP/earth2/multisenge_utae/artifacts/slurm-%j.err

# No set -u: PARAM module purge triggers conda deactivate "unbound variable" errors.
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
  --epochs 80 \
  --batch-size 2 \
  --workers 2 \
  --out-dir multisenge_utae/checkpoints/run_c6_head_v0
