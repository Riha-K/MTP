#!/bin/bash
# Stage 1: ConvNeXt-Tiny student aligned to the frozen GELU S2-only U-TAE.
# Does not touch cmu_s1_vit_v1, cmu_s1_vit_gelu_v0, or the SiLU U-TAE run.
#   cd /home/rihak_iitp/MTP/earth2
#   sbatch --exclude=ragpu004,ragpu005,ragpu007 multisenge_utae/train_cmu_convnext_gelu.sbatch
#SBATCH -J cmu_s1_cnx
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 48:00:00
#SBATCH -o /home/rihak_iitp/MTP/earth2/multisenge_utae/artifacts/slurm-cmu-cnx-gelu-%j.out
#SBATCH -e /home/rihak_iitp/MTP/earth2/multisenge_utae/artifacts/slurm-cmu-cnx-gelu-%j.err

set -eo pipefail
cd /home/rihak_iitp/MTP/earth2
mkdir -p multisenge_utae/artifacts multisenge_utae/checkpoints

TEACHER="${TEACHER:-multisenge_utae/checkpoints/run_c10_s2_gelu_full_v0/best.pt}"
OUT="${OUT:-multisenge_utae/checkpoints/cmu_s1_convnext_gelu_v0}"
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
  --student convnext \
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
