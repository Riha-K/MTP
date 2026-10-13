#!/bin/bash
# Stage 2: 10-class CONCAT full, from the ReLU U-TAE SAR head.
# Does not touch cmu_vit_c10_full_v0 or cmu_convnext_c10_full_v0.
#   cd /home/rihak_iitp/MTP/earth2
#   sbatch --dependency=afterok:109564 --exclude=ragpu004,ragpu005,ragpu007 \
#     multisenge_utae/train_cmu_utae_c10_full.sbatch
#SBATCH -J cmu_utae_c10f
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH -o /home/rihak_iitp/MTP/earth2/multisenge_utae/artifacts/slurm-cmu-utae-c10-full-%j.out
#SBATCH -e /home/rihak_iitp/MTP/earth2/multisenge_utae/artifacts/slurm-cmu-utae-c10-full-%j.err

set -eo pipefail
cd /home/rihak_iitp/MTP/earth2
mkdir -p multisenge_utae/artifacts multisenge_utae/checkpoints

HEAD="${HEAD:-multisenge_utae/checkpoints/cmu_utae_c10_head_v0/best.pt}"
OUT=multisenge_utae/checkpoints/cmu_utae_c10_full_v0
if [[ ! -f "$HEAD" ]]; then
  echo "missing head ckpt: $HEAD"
  exit 1
fi
if [[ -f "$OUT/last.pt" ]]; then
  echo "refusing to overwrite $OUT/last.pt"
  exit 1
fi

module purge
module load MLDL/Pytorch-gpu
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

python -m multisenge_utae.train_cmu_vit \
  --index multisenge_seg/artifacts/patch_index.json \
  --data-root LULCDial-s1/data/lulcdial_s1/ai4lcc/multisenge \
  --num-classes 10 \
  --mode full \
  --fusion concat \
  --student utae \
  --activation relu \
  --seed 42 \
  --init-ckpt "$HEAD" \
  --epochs 80 \
  --batch-size 2 \
  --workers 2 \
  --patience 20 \
  --out-dir "$OUT"

python -m multisenge_utae.plot_history \
  "$OUT/history.json" \
  --title "CMU U-TAE SAR P5 full ReLU (10-class)" \
  || echo "plot skipped"
