#!/bin/bash
# Phase 5: 31UEQ test for 10c CONCAT CE+GDL full.
#   sbatch --dependency=afterok:FULL --exclude=ragpu004,ragpu005,ragpu007 \
#     multisenge_utae/eval_cmu_utae_c10_cegdl_full.sbatch
#SBATCH -J cmu_cegdfe
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 02:00:00
#SBATCH -o /home/rihak_iitp/MTP/earth2/multisenge_utae/artifacts/slurm-cmu-utae-c10-cegdl-full-eval-%j.out
#SBATCH -e /home/rihak_iitp/MTP/earth2/multisenge_utae/artifacts/slurm-cmu-utae-c10-cegdl-full-eval-%j.err

set -eo pipefail
cd /home/rihak_iitp/MTP/earth2

CKPT="${CKPT:-multisenge_utae/checkpoints/cmu_utae_c10_cegdl_full_v0/best.pt}"
OUT="${OUT:-multisenge_utae/results/concat_utae/cmu_utae_c10_cegdl_full_v0}"
mkdir -p  multisenge_utae/artifacts "${OUT}"

if [[ ! -f "$CKPT" ]]; then
  echo "missing checkpoint: $CKPT"
  exit 1
fi

module purge
module load MLDL/Pytorch-gpu
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

echo "eval CKPT=${CKPT}"
echo "eval OUT=${OUT}"

python -m  multisenge_utae.train_cmu_vit \
  --index  multisenge_seg/artifacts/patch_index.json \
  --data-root LULCDial-s1/data/lulcdial_s1/ai4lcc/multisenge \
  --eval-ckpt "${CKPT}" \
  --eval-split test \
  --num-classes 10 \
  --fusion concat \
  --student utae \
  --activation relu \
  --batch-size 2 \
  --workers 2 \
  --out-dir "${OUT}"

python -m  multisenge_utae.export_notes \
  --metrics "${OUT}/test_metrics.json" \
  --num-classes 10 \
  --title "CMU U-TAE concat full CE+GDL 10-class test (31UEQ)" \
  --out "${OUT}/test_metrics.md"
