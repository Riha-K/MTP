#!/bin/bash
# Phase 5 Stage 2: 10-class CONCAT head, CE+GDL. New Stage 1 student + CE+GDL S2 teacher.
# Does not touch cmu_utae_c10_head_v0.
#   sbatch --dependency=afterok:113573 --exclude=ragpu004,ragpu005,ragpu007 \
#     multisenge_utae/train_cmu_utae_c10_cegdl_head.sbatch
#SBATCH -J cmu_cegdh
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH -o /home/rihak_iitp/MTP/earth2/multisenge_utae/artifacts/slurm-cmu-utae-c10-cegdl-head-%j.out
#SBATCH -e /home/rihak_iitp/MTP/earth2/multisenge_utae/artifacts/slurm-cmu-utae-c10-cegdl-head-%j.err

set -eo pipefail
cd /home/rihak_iitp/MTP/earth2
mkdir -p  multisenge_utae/artifacts  multisenge_utae/checkpoints

STUDENT="${STUDENT:-multisenge_utae/checkpoints/cmu_s1_utae_cegdl_v0/student_best.pt}"
S2="${S2:-multisenge_utae/checkpoints/run_c10_s2_cegdl_full_v0/best.pt}"
OUT=multisenge_utae/checkpoints/cmu_utae_c10_cegdl_head_v0
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

python -m  multisenge_utae.train_cmu_vit \
  --index  multisenge_seg/artifacts/patch_index.json \
  --data-root LULCDial-s1/data/lulcdial_s1/ai4lcc/multisenge \
  --num-classes 10 \
  --mode head \
  --fusion concat \
  --student utae \
  --activation relu \
  --loss ce_gdl \
  --dice-lambda 1 \
  --seed 42 \
  --student-ckpt "$STUDENT" \
  --s2-ckpt "$S2" \
  --epochs 80 \
  --batch-size 2 \
  --workers 2 \
  --patience 20 \
  --out-dir "$OUT"

python -m  multisenge_utae.plot_history \
  "$OUT/history.json" \
  --title "CMU U-TAE SAR P4 head CE+GDL (10-class)" \
  || echo "plot skipped"
