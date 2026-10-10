#!/bin/bash
# 10-class logit residual. Frozen ReLU S2-only full model.
# Frozen Stage 1 student is cmu_s1_vit_v1 (val retrieval 0.416).
# The SAR correction starts at zero. New folder. Does not touch the running
# extra-negative Stage 1 job or the finished photo-only checkpoint.
#   cd /home/rihak_iitp/MTP/earth2
#   sbatch --exclude=ragpu004,ragpu005,ragpu007 multisenge_utae/train_residual_c10.sbatch
#SBATCH -J c10_resid
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH -o /home/rihak_iitp/MTP/earth2/multisenge_utae/artifacts/slurm-c10-resid-%j.out
#SBATCH -e /home/rihak_iitp/MTP/earth2/multisenge_utae/artifacts/slurm-c10-resid-%j.err

set -eo pipefail
cd /home/rihak_iitp/MTP/earth2
mkdir -p multisenge_utae/artifacts multisenge_utae/checkpoints

module purge
module load MLDL/Pytorch-gpu
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

S2="${S2:-multisenge_utae/checkpoints/run_c10_s2_full_v0/best.pt}"
STUDENT="${STUDENT:-multisenge_utae/checkpoints/cmu_s1_vit_v1/student_best.pt}"
OUT="${OUT:-multisenge_utae/checkpoints/c10_logit_residual_v0}"
if [[ ! -f "$S2" ]]; then
  echo "missing S2 checkpoint: $S2"
  exit 1
fi
if [[ ! -f "$STUDENT" ]]; then
  echo "missing Stage 1 student: $STUDENT"
  exit 1
fi
if [[ -f "$OUT/last.pt" ]]; then
  echo "refusing to overwrite $OUT (last.pt already exists)"
  exit 1
fi

python -m multisenge_utae.train_residual \
  --index multisenge_seg/artifacts/patch_index.json \
  --data-root LULCDial-s1/data/lulcdial_s1/ai4lcc/multisenge \
  --num-classes 10 \
  --s2-ckpt "$S2" \
  --student-ckpt "$STUDENT" \
  --epochs 80 \
  --batch-size 2 \
  --lr 1e-4 \
  --workers 2 \
  --patience 20 \
  --seed 42 \
  --out-dir "$OUT"

python -m multisenge_utae.plot_history \
  "$OUT/history.json" \
  --title "10-class logit residual" \
  || echo "plot skipped"
