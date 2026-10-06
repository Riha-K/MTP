#!/bin/bash
# Phase 4 step 2: test the GELU S2-only full teacher on tile 31UEQ.
# Keep it only if weighted F1 / kappa is at least 0.8865 / 0.7945.
#   sbatch --dependency=afterok:FULL_JOB --exclude=ragpu004,ragpu005,ragpu007 multisenge_utae/eval_s2_c10_gelu_full.sbatch
#SBATCH -J utae_s2_c10_gelu_ev
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 02:00:00
#SBATCH -o /home/rihak_iitp/MTP/earth2/multisenge_utae/artifacts/slurm-s2-c10-gelu-eval-%j.out
#SBATCH -e /home/rihak_iitp/MTP/earth2/multisenge_utae/artifacts/slurm-s2-c10-gelu-eval-%j.err

set -eo pipefail
cd /home/rihak_iitp/MTP/earth2

CKPT="${CKPT:-multisenge_utae/checkpoints/run_c10_s2_gelu_full_v0/best.pt}"
OUT="${OUT:-multisenge_utae/results/concat_utae/run_c10_s2_gelu_full_v0}"
mkdir -p multisenge_utae/artifacts "${OUT}"

module purge
module load MLDL/Pytorch-gpu
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

echo "eval CKPT=${CKPT}"
echo "eval OUT=${OUT}"

python -m multisenge_utae.train \
  --index multisenge_seg/artifacts/patch_index.json \
  --eval-ckpt "${CKPT}" \
  --eval-split test \
  --num-classes 10 \
  --batch-size 2 \
  --out-dir "${OUT}"

python -m multisenge_utae.export_notes \
  --metrics "${OUT}/test_metrics.json" \
  --num-classes 10 \
  --title "U-TAE S2-only P5 full GELU 10-class test (31UEQ)" \
  --out "${OUT}/test_metrics.md"
