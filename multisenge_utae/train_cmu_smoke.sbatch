#!/bin/bash
# Tiny CMU smoke on PARAM (few patches, 1 epoch).
#   sbatch --exclude=ragpu004,ragpu005,ragpu007 multisenge_utae/train_cmu_smoke.sbatch
#SBATCH -J cmu_smoke
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 02:00:00
#SBATCH -o /home/rihak_iitp/MTP/earth2/multisenge_utae/logs/slurm-cmu-smoke-%j.out
#SBATCH -e /home/rihak_iitp/MTP/earth2/multisenge_utae/logs/slurm-cmu-smoke-%j.err

set -eo pipefail
cd /home/rihak_iitp/MTP/earth2
mkdir -p multisenge_utae/logs multisenge_utae/checkpoints

module purge
module load MLDL/Pytorch-gpu
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

TEACHER="${TEACHER:-multisenge_utae/checkpoints/run_c10_s2_full_v0/best.pt}"
if [[ ! -f "$TEACHER" ]]; then
  echo "missing teacher ckpt: $TEACHER"
  exit 1
fi

python -m multisenge_utae.train_cmu \
  --index multisenge_seg/artifacts/patch_index.json \
  --teacher-ckpt "$TEACHER" \
  --epochs 1 \
  --batch-size 2 \
  --max-train 8 \
  --max-val 4 \
  --workers 0 \
  --out-dir multisenge_utae/checkpoints/cmu_s1_vit_smoke
