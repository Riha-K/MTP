#!/bin/bash
# Quick P3 smoke (~10-20 min): tiny patch subsample, same ckpt path as full probe.
#   sbatch multisenge_utae/probe_smoke.sbatch
#SBATCH -J utae_probe_smoke
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 00:45:00
#SBATCH -o /home/rihak_iitp/MTP/earth2/multisenge_utae/artifacts/slurm-probe-smoke-%j.out
#SBATCH -e /home/rihak_iitp/MTP/earth2/multisenge_utae/artifacts/slurm-probe-smoke-%j.err

set -eo pipefail
cd /home/rihak_iitp/MTP/earth2
mkdir -p multisenge_utae/artifacts multisenge_utae/results/probe_c6_smoke

module purge
module load MLDL/Pytorch-gpu
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

CKPT="${CKPT:-multisenge_utae/checkpoints/run_c6_head_v0/best.pt}"

python -m multisenge_utae.probe_layers \
  --index multisenge_seg/artifacts/patch_index.json \
  --num-classes 6 \
  --probe linear \
  --ckpt "${CKPT}" \
  --batch-size 1 \
  --workers 0 \
  --max-train-patches 24 \
  --max-val-patches 12 \
  --pixels-per-patch 256 \
  --out-dir multisenge_utae/results/probe_c6_smoke
