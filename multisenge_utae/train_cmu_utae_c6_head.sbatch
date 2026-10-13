#!/bin/bash
# Stage 2: 6-class CONCAT head. Same ReLU S2 teacher and ReLU U-TAE student as the 10-class head.
# The 6-class and 10-class heads share the 10-class S2 encoder.
# Does not touch cmu_vit_c6_head_v0 or cmu_convnext_c6_head_v0.
#   cd /home/rihak_iitp/MTP/earth2
#   sbatch --dependency=afterok:109556 --exclude=ragpu004,ragpu005,ragpu007 \
#     multisenge_utae/train_cmu_utae_c6_head.sbatch
#SBATCH -J cmu_utae_c6h
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH -o /home/rihak_iitp/MTP/earth2/multisenge_utae/artifacts/slurm-cmu-utae-c6-head-%j.out
#SBATCH -e /home/rihak_iitp/MTP/earth2/multisenge_utae/artifacts/slurm-cmu-utae-c6-head-%j.err

set -eo pipefail
cd /home/rihak_iitp/MTP/earth2
mkdir -p multisenge_utae/artifacts multisenge_utae/checkpoints

STUDENT="${STUDENT:-multisenge_utae/checkpoints/cmu_s1_utae_relu_v0/student_best.pt}"
S2="${S2:-multisenge_utae/checkpoints/run_c10_s2_full_v0/best.pt}"
OUT=multisenge_utae/checkpoints/cmu_utae_c6_head_v0
if [[ ! -f "$STUDENT" ]]; then
  echo "missing Stage1 student: $STUDENT"
  exit 1
fi
if [[ ! -f "$S2" ]]; then
  echo "missing S2 teacher: $S2"
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
  --num-classes 6 \
  --mode head \
  --fusion concat \
  --student utae \
  --activation relu \
  --seed 42 \
  --student-ckpt "$STUDENT" \
  --s2-ckpt "$S2" \
  --epochs 80 \
  --batch-size 2 \
  --workers 2 \
  --patience 20 \
  --out-dir "$OUT"

python -m multisenge_utae.plot_history \
  "$OUT/history.json" \
  --title "CMU U-TAE SAR P4 head ReLU (6-class)" \
  || echo "plot skipped"
