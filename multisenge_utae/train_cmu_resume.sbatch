#!/bin/bash
# Continue Stage 1 from cmu_s1_vit_v1/last.pt (epoch 80, lr already 0).
# New cosine peaks at 3e-5 for epochs 81-160. Patience 20 stops the job
# if validation accuracy does not beat 0.416. best.pt stays epoch 70 until then.
#   sbatch --exclude=ragpu004,ragpu005,ragpu007 multisenge_utae/train_cmu_resume.sbatch
#SBATCH -J cmu_s1_r
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH -o /home/rihak_iitp/MTP/earth2/multisenge_utae/artifacts/slurm-cmu-%j.out
#SBATCH -e /home/rihak_iitp/MTP/earth2/multisenge_utae/artifacts/slurm-cmu-%j.err

set -eo pipefail
cd /home/rihak_iitp/MTP/earth2
mkdir -p multisenge_utae/artifacts multisenge_utae/checkpoints multisenge_utae/logs

module purge
module load MLDL/Pytorch-gpu
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

TEACHER="${TEACHER:-multisenge_utae/checkpoints/run_c10_s2_full_v0/best.pt}"
OUT="${OUT:-multisenge_utae/checkpoints/cmu_s1_vit_v1}"
RESUME="${RESUME:-$OUT/last.pt}"
if [[ ! -f "$TEACHER" ]]; then
  echo "missing teacher ckpt: $TEACHER"
  exit 1
fi
if [[ ! -f "$RESUME" ]]; then
  echo "missing resume ckpt: $RESUME"
  exit 1
fi

python -m multisenge_utae.train_cmu \
  --index multisenge_seg/artifacts/patch_index.json \
  --data-root LULCDial-s1/data/lulcdial_s1/ai4lcc/multisenge \
  --teacher-ckpt "$TEACHER" \
  --resume "$RESUME" \
  --epochs 160 \
  --batch-size 2 \
  --lr 3e-5 \
  --temperature 0.07 \
  --proj-dim 256 \
  --workers 2 \
  --patience 20 \
  --seed 42 \
  --out-dir "$OUT"

echo "CMU resume done. student_best.pt -> $OUT/student_best.pt"
