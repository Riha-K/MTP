#!/bin/bash
# PARAM: P4 head-only 10-class (must finish before P5 full)
#   sbatch --exclude=ragpu004,ragpu005,ragpu007 multisenge_utae/train_c10_head.sbatch
#SBATCH -J utae_c10_head
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH -o /home/rihak_iitp/MTP/earth2/multisenge_utae/artifacts/slurm-c10-head-%j.out
#SBATCH -e /home/rihak_iitp/MTP/earth2/multisenge_utae/artifacts/slurm-c10-head-%j.err

set -eo pipefail
cd /home/rihak_iitp/MTP/earth2
mkdir -p multisenge_utae/artifacts multisenge_utae/checkpoints

module purge
module load MLDL/Pytorch-gpu
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

python -m multisenge_utae.train \
  --index multisenge_seg/artifacts/patch_index.json \
  --data-root LULCDial-s1/data/lulcdial_s1/ai4lcc/multisenge \
  --num-classes 10 \
  --mode head \
  --epochs 80 \
  --batch-size 2 \
  --workers 2 \
  --out-dir multisenge_utae/checkpoints/run_c10_head_v0

python -m multisenge_utae.plot_history \
  multisenge_utae/checkpoints/run_c10_head_v0/history.json \
  --title "U-TAE P4 head-only (10-class)" \
  --out multisenge_utae/checkpoints/run_c10_head_v0/history_plot.png \
  || echo "plot skipped (install matplotlib if needed)"
