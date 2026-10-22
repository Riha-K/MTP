#!/bin/bash
# Phase 5: 31UEQ test for Composite B S2-only full.
#   sbatch --dependency=afterok:111850 --exclude=ragpu004,ragpu005,ragpu007  multisenge_utae/eval_s2_c10_cegdl_full.sbatch
#SBATCH -J s2e_cegdf
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 02:00:00
#SBATCH -o /home/rihak_iitp/MTP/earth2/multisenge_utae/artifacts/slurm-s2-c10-cegdl-full-eval-%j.out
#SBATCH -e /home/rihak_iitp/MTP/earth2/multisenge_utae/artifacts/slurm-s2-c10-cegdl-full-eval-%j.err

set -eo pipefail
cd /home/rihak_iitp/MTP/earth2

CKPT="${CKPT:-multisenge_utae/checkpoints/run_c10_s2_cegdl_full_v0/best.pt}"
OUT="${OUT:-multisenge_utae/results/concat_utae/run_c10_s2_cegdl_full_v0}"
mkdir -p  multisenge_utae/artifacts "${OUT}"

module purge
module load MLDL/Pytorch-gpu
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

echo "eval CKPT=${CKPT}"
echo "eval OUT=${OUT}"

python -m  multisenge_utae.train \
  --index  multisenge_seg/artifacts/patch_index.json \
  --eval-ckpt "${CKPT}" \
  --eval-split test \
  --num-classes 10 \
  --batch-size 2 \
  --out-dir "${OUT}"

python -m  multisenge_utae.export_notes \
  --metrics "${OUT}/test_metrics.json" \
  --num-classes 10 \
  --title "U-TAE S2-only full CE+GDL 10-class test (31UEQ)" \
  --out "${OUT}/test_metrics.md"
