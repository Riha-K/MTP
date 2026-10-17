#!/bin/bash
# Stage 2: 10-class CONCAT head for SEED 43-46. No 6-class job.
#SBATCH -J u10h_seed
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH -o /home/rihak_iitp/MTP/earth2/multisenge_utae/artifacts/slurm-cmu-utae-c10-head-seed-%j.out
#SBATCH -e /home/rihak_iitp/MTP/earth2/multisenge_utae/artifacts/slurm-cmu-utae-c10-head-seed-%j.err

set -eo pipefail
cd /home/rihak_iitp/MTP/earth2
if [[ -z "${SEED:-}" || "$SEED" == "42" ]]; then
  echo "set SEED to 43-46; seed 42 stays in cmu_utae_c10_head_v0"
  exit 1
fi
STUDENT="${STUDENT:-multisenge_utae/checkpoints/cmu_s1_utae_relu_s${SEED}_v0/student_best.pt}"
S2="${S2:-multisenge_utae/checkpoints/run_c10_s2_full_s${SEED}_v0/best.pt}"
OUT="multisenge_utae/checkpoints/cmu_utae_c10_head_s${SEED}_v0"
mkdir -p multisenge_utae/artifacts "$OUT"
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
  --num-classes 10 \
  --mode head \
  --fusion concat \
  --student utae \
  --activation relu \
  --seed "$SEED" \
  --student-ckpt "$STUDENT" \
  --s2-ckpt "$S2" \
  --epochs 80 \
  --batch-size 2 \
  --workers 2 \
  --patience 20 \
  --out-dir "$OUT"

python -m multisenge_utae.plot_history \
  "$OUT/history.json" \
  --title "CMU U-TAE SAR P4 head ReLU seed ${SEED} (10-class)" \
  || echo "plot skipped"
