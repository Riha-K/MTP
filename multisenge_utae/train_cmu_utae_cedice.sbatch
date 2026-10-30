#!/bin/bash
# Phase 5 Stage 1: ReLU U-TAE SAR student vs frozen CE+Dice S2 teacher.
# Request 48h for 160 epochs. If Slurm rejects MaxTime, resubmit with -t 24:00:00.
#   sbatch --dependency=afterok:113571 --exclude=ragpu004,ragpu005,ragpu007  multisenge_utae/train_cmu_utae_cedice.sbatch
#SBATCH -J s1u_cedice
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 48:00:00
#SBATCH -o /home/rihak_iitp/MTP/earth2/multisenge_utae/artifacts/slurm-cmu-utae-cedice-%j.out
#SBATCH -e /home/rihak_iitp/MTP/earth2/multisenge_utae/artifacts/slurm-cmu-utae-cedice-%j.err

set -eo pipefail
cd /home/rihak_iitp/MTP/earth2
TEACHER="${TEACHER:-multisenge_utae/checkpoints/run_c10_s2_cedice_full_v0/best.pt}"
OUT="${OUT:-multisenge_utae/checkpoints/cmu_s1_utae_cedice_v0}"
mkdir -p  multisenge_utae/artifacts "$OUT"
if [[ ! -f "$TEACHER" ]]; then
  echo "missing teacher ckpt: $TEACHER"
  exit 1
fi
if [[ -f "$OUT/last.pt" ]]; then
  echo "refusing to overwrite $OUT/last.pt; use the resume sbatch"
  exit 1
fi

module purge
module load MLDL/Pytorch-gpu
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

python -m  multisenge_utae.train_cmu \
  --index  multisenge_seg/artifacts/patch_index.json \
  --data-root LULCDial-s1/data/lulcdial_s1/ai4lcc/multisenge \
  --teacher-ckpt "$TEACHER" \
  --student utae \
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
