#!/bin/bash
# Continue Phase 5 Stage 1 CE+Dice student if the first 48h slice is killed.
# If history.json exists the first job finished cleanly and this exits 0.
# Use afterany: SIGKILL on the first slice would skip afterok.
#   sbatch --dependency=afterany:FIRST_S1 --exclude=ragpu004,ragpu005,ragpu007  multisenge_utae/train_cmu_utae_cedice_resume.sbatch
#SBATCH -J s1u_cediceR
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 48:00:00
#SBATCH -o /home/rihak_iitp/MTP/earth2/multisenge_utae/artifacts/slurm-cmu-utae-cedice-resume-%j.out
#SBATCH -e /home/rihak_iitp/MTP/earth2/multisenge_utae/artifacts/slurm-cmu-utae-cedice-resume-%j.err

set -eo pipefail
cd /home/rihak_iitp/MTP/earth2
TEACHER="${TEACHER:-multisenge_utae/checkpoints/run_c10_s2_cedice_full_v0/best.pt}"
OUT="${OUT:-multisenge_utae/checkpoints/cmu_s1_utae_cedice_v0}"
mkdir -p  multisenge_utae/artifacts "$OUT"
if [[ -f "$OUT/history.json" ]]; then
  echo "already finished: $OUT/history.json"
  exit 0
fi
if [[ ! -f "$OUT/last.pt" ]]; then
  echo "missing $OUT/last.pt"
  exit 1
fi
if [[ ! -f "$TEACHER" ]]; then
  echo "missing teacher ckpt: $TEACHER"
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
  --resume "$OUT/last.pt" \
  --out-dir "$OUT"

echo "CMU resume done. student_best.pt -> $OUT/student_best.pt"
