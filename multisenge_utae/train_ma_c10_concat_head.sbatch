#!/bin/bash
# Task M ablation: MA-UTAE concat fuse 10c P4 head.
#   sbatch --exclude=ragpu004,ragpu005,ragpu007 multisenge_utae/train_ma_c10_concat_head.sbatch
#SBATCH -J ma_c10_c_head
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH -o /home/rihak_iitp/MTP/earth2/multisenge_utae/artifacts/slurm-ma-c10-c-head-%j.out
#SBATCH -e /home/rihak_iitp/MTP/earth2/multisenge_utae/artifacts/slurm-ma-c10-c-head-%j.err

set -eo pipefail
cd /home/rihak_iitp/MTP/earth2
mkdir -p multisenge_utae/artifacts multisenge_utae/checkpoints

module purge
module load MLDL/Pytorch-gpu
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

python -m multisenge_utae.train_ma \
  --index multisenge_seg/artifacts/patch_index.json \
  --data-root LULCDial-s1/data/lulcdial_s1/ai4lcc/multisenge \
  --num-classes 10 \
  --mode head \
  --fusion concat \
  --epochs 80 \
  --batch-size 2 \
  --workers 2 \
  --out-dir multisenge_utae/checkpoints/ma_c10_concat_head_v0

python -m multisenge_utae.plot_history \
  multisenge_utae/checkpoints/ma_c10_concat_head_v0/history.json \
  --title "MA-UTAE concat-fuse P4 head (10-class)" \
  || echo "plot skipped"
