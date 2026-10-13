#!/bin/bash
# Stage 1: ReLU U-TAE SAR student aligned to the frozen ReLU S2-only U-TAE.
# Does not retrain the teacher. Does not touch the ViT or ConvNeXt students.
#   cd /home/rihak_iitp/MTP/earth2
#   sbatch --exclude=ragpu004,ragpu005,ragpu007 multisenge_utae/train_cmu_utae_relu.sbatch
#SBATCH -J cmu_s1_utae
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 48:00:00
#SBATCH -o /home/rihak_iitp/MTP/earth2/multisenge_utae/artifacts/slurm-cmu-utae-relu-%j.out
#SBATCH -e /home/rihak_iitp/MTP/earth2/multisenge_utae/artifacts/slurm-cmu-utae-relu-%j.err

set -eo pipefail
cd /home/rihak_iitp/MTP/earth2
mkdir -p multisenge_utae/artifacts multisenge_utae/checkpoints

TEACHER="${TEACHER:-multisenge_utae/checkpoints/run_c10_s2_full_v0/best.pt}"
OUT="${OUT:-multisenge_utae/checkpoints/cmu_s1_utae_relu_v0}"
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
  --epochs 80 \
  --batch-size 2 \
  --lr 3e-4 \
  --temperature 0.07 \
  --proj-dim 256 \
  --workers 2 \
  --patience 20 \
  --seed 42 \
  --out-dir "$OUT"

echo "CMU done. student_best.pt -> $OUT/student_best.pt"
