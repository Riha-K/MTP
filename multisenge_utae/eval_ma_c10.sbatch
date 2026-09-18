#!/bin/bash
# PARAM: MA-UTAE 10-class test eval (gated or concat — fusion read from ckpt meta).
#   CKPT=multisenge_utae/checkpoints/ma_c10_gated_full_v0/best.pt \
#   OUT=multisenge_utae/results/ma_utae/ma_c10_gated_full_v0 \
#   sbatch --exclude=ragpu004,ragpu005,ragpu007 multisenge_utae/eval_ma_c10.sbatch
#SBATCH -J ma_c10_eval
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 02:00:00
#SBATCH -o /home/rihak_iitp/MTP/earth2/multisenge_utae/artifacts/slurm-ma-c10-eval-%j.out
#SBATCH -e /home/rihak_iitp/MTP/earth2/multisenge_utae/artifacts/slurm-ma-c10-eval-%j.err

set -eo pipefail
cd /home/rihak_iitp/MTP/earth2

CKPT="${CKPT:-multisenge_utae/checkpoints/ma_c10_gated_full_v0/best.pt}"
OUT="${OUT:-multisenge_utae/results/ma_utae/ma_c10_gated_full_v0}"
mkdir -p multisenge_utae/artifacts "${OUT}"

module purge
module load MLDL/Pytorch-gpu
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

echo "eval CKPT=${CKPT}"
echo "eval OUT=${OUT}"

python -m multisenge_utae.train_ma \
  --index multisenge_seg/artifacts/patch_index.json \
  --eval-ckpt "${CKPT}" \
  --eval-split test \
  --num-classes 10 \
  --batch-size 2 \
  --out-dir "${OUT}"

python -m multisenge_utae.export_notes \
  --metrics "${OUT}/test_metrics.json" \
  --num-classes 10 \
  --title "MA-UTAE 10-class test (31UEQ)" \
  --out "${OUT}/test_metrics.md"
