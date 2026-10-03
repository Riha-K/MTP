#!/bin/bash
# Test tile 31UEQ for the finished CMU-ViT U-TAE 10c P5 checkpoint.
# Submit only after 107114 writes best.pt:
#   sbatch --dependency=afterok:107114 --exclude=ragpu004,ragpu005,ragpu007 multisenge_utae/eval_cmu_vit_c10_full.sbatch
#SBATCH -J cmuvit_c10fe
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 02:00:00
#SBATCH -o /home/rihak_iitp/MTP/earth2/multisenge_utae/logs/slurm-cmuvit-c10-full-eval-%j.out
#SBATCH -e /home/rihak_iitp/MTP/earth2/multisenge_utae/logs/slurm-cmuvit-c10-full-eval-%j.err

set -eo pipefail
cd /home/rihak_iitp/MTP/earth2

CKPT="${CKPT:-multisenge_utae/checkpoints/cmu_vit_c10_full_v0/best.pt}"
OUT="${OUT:-multisenge_utae/results/cmu_vit/cmu_vit_c10_full_v0}"
NUM_CLASSES="${NUM_CLASSES:-10}"
mkdir -p multisenge_utae/logs "${OUT}"

if [[ ! -f "$CKPT" ]]; then
  echo "missing checkpoint: $CKPT"
  exit 1
fi

module purge
module load MLDL/Pytorch-gpu
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

echo "eval CKPT=${CKPT}"
echo "eval OUT=${OUT}"
echo "eval NUM_CLASSES=${NUM_CLASSES}"

python -m multisenge_utae.train_cmu_vit \
  --index multisenge_seg/artifacts/patch_index.json \
  --data-root LULCDial-s1/data/lulcdial_s1/ai4lcc/multisenge \
  --eval-ckpt "${CKPT}" \
  --eval-split test \
  --num-classes "${NUM_CLASSES}" \
  --batch-size 2 \
  --workers 2 \
  --out-dir "${OUT}"

python -m multisenge_utae.export_notes \
  --metrics "${OUT}/test_metrics.json" \
  --num-classes "${NUM_CLASSES}" \
  --title "CMU-ViT U-TAE 10-class P5 test (31UEQ)" \
  --out "${OUT}/test_metrics.md"
