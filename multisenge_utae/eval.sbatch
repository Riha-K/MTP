#!/bin/bash
#SBATCH -J msge_utae_eval
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 02:00:00
#SBATCH -o /home/rihak_iitp/MTP/earth2/multisenge_utae/artifacts/slurm-eval-%j.out
#SBATCH -e /home/rihak_iitp/MTP/earth2/multisenge_utae/artifacts/slurm-eval-%j.err

set -euo pipefail
cd /home/rihak_iitp/MTP/earth2
mkdir -p multisenge_utae/artifacts multisenge_utae/results/run_c6_head_v0

module purge
module load MLDL/Pytorch-gpu
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

CKPT="${CKPT:-multisenge_utae/checkpoints/run_c6_head_v0/best.pt}"
OUT="multisenge_utae/results/run_c6_head_v0"

python -m multisenge_utae.train \
  --index multisenge_seg/artifacts/patch_index.json \
  --eval-ckpt "${CKPT}" \
  --eval-split test \
  --batch-size 2 \
  --out-dir "${OUT}"

python -m multisenge_utae.export_notes \
  --metrics "${OUT}/test_metrics.json" \
  --num-classes 6 \
  --title "U-TAE 6-class test (31UEQ)" \
  --out "${OUT}/test_metrics.md"
