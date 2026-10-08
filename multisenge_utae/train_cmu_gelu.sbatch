#!/bin/bash
# Phase 4 Stage 1: ViT-B/16 S1 student, frozen GELU S2-only full teacher.
# Teacher is the spatial encoder only (no L-TAE). Writes cmu_s1_vit_gelu_v0.
# Does not touch checkpoints/cmu_s1_vit_v1/student_best.pt.
# best.pt is already the epoch-37 file. No dependency on 108194.
#   sbatch --exclude=ragpu004,ragpu005,ragpu007 multisenge_utae/train_cmu_gelu.sbatch
#SBATCH -J cmu_s1_vit_gelu
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 48:00:00
#SBATCH -o /home/rihak_iitp/MTP/earth2/multisenge_utae/artifacts/slurm-cmu-gelu-%j.out
#SBATCH -e /home/rihak_iitp/MTP/earth2/multisenge_utae/artifacts/slurm-cmu-gelu-%j.err

set -eo pipefail
cd /home/rihak_iitp/MTP/earth2
mkdir -p multisenge_utae/artifacts multisenge_utae/checkpoints

module purge
module load MLDL/Pytorch-gpu
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

TEACHER="${TEACHER:-multisenge_utae/checkpoints/run_c10_s2_gelu_full_v0/best.pt}"
if [[ ! -f "$TEACHER" ]]; then
  echo "missing teacher ckpt: $TEACHER"
  exit 1
fi

OUT="${OUT:-multisenge_utae/checkpoints/cmu_s1_vit_gelu_v0}"

python -m multisenge_utae.train_cmu \
  --index multisenge_seg/artifacts/patch_index.json \
  --data-root LULCDial-s1/data/lulcdial_s1/ai4lcc/multisenge \
  --teacher-ckpt "$TEACHER" \
  --epochs 160 \
  --batch-size 2 \
  --lr 3e-4 \
  --temperature 0.07 \
  --proj-dim 256 \
  --workers 2 \
  --patience 20 \
  --seed 42 \
  --out-dir "$OUT"

echo "CMU done. student_best.pt -> $OUT/student_best.pt"
