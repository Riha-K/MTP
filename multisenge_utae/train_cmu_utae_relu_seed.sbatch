#!/bin/bash
# Stage 1: ReLU U-TAE SAR student, 160 epochs. Set SEED (43-46).
# Teacher is that seed's S2 full. Does not touch cmu_s1_utae_relu_v0.
#SBATCH -J s1u_seed
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 48:00:00
#SBATCH -o /home/rihak_iitp/MTP/earth2/multisenge_utae/artifacts/slurm-cmu-utae-relu-seed-%j.out
#SBATCH -e /home/rihak_iitp/MTP/earth2/multisenge_utae/artifacts/slurm-cmu-utae-relu-seed-%j.err

set -eo pipefail
cd /home/rihak_iitp/MTP/earth2
if [[ -z "${SEED:-}" || "$SEED" == "42" ]]; then
  echo "set SEED to 43-46; seed 42 stays in cmu_s1_utae_relu_v0"
  exit 1
fi
TEACHER="${TEACHER:-multisenge_utae/checkpoints/run_c10_s2_full_s${SEED}_v0/best.pt}"
OUT="${OUT:-multisenge_utae/checkpoints/cmu_s1_utae_relu_s${SEED}_v0}"
mkdir -p multisenge_utae/artifacts "$OUT"
if [[ ! -f "$TEACHER" ]]; then
  echo "missing teacher ckpt: $TEACHER"
  exit 1
fi
if [[ -f "$OUT/last.pt" ]]; then
  echo "refusing to overwrite $OUT/last.pt"
  exit 1
fi

module purge
module load MLDL/Pytorch-gpu
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

python -m multisenge_utae.train_cmu \
  --index multisenge_seg/artifacts/patch_index.json \
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
  --seed "$SEED" \
  --out-dir "$OUT"

echo "CMU done. student_best.pt -> $OUT/student_best.pt"
