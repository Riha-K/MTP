#!/bin/bash
# Continue ConvNeXt Stage 1 from cmu_s1_convnext_gelu_v0/last.pt.
# Same step as the ReLU ViT resume: epochs through 160, new cosine, lr 3e-5.
# Patience 20. best.pt changes only when val retrieval acc strictly increases.
# Waits until job 108993 exits 0 and the SiLU head test 109029 has finished,
# so this does not take a GPU while those two still need one.
#   cd /home/rihak_iitp/MTP/earth2
#   sbatch --dependency=afterok:108993,afterany:109029 --exclude=ragpu004,ragpu005,ragpu007 multisenge_utae/train_cmu_convnext_gelu_resume.sbatch
#SBATCH -J cmu_s1_cnx_r
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 48:00:00
#SBATCH -o /home/rihak_iitp/MTP/earth2/multisenge_utae/artifacts/slurm-cmu-cnx-gelu-resume-%j.out
#SBATCH -e /home/rihak_iitp/MTP/earth2/multisenge_utae/artifacts/slurm-cmu-cnx-gelu-resume-%j.err

set -eo pipefail
cd /home/rihak_iitp/MTP/earth2
mkdir -p multisenge_utae/artifacts multisenge_utae/checkpoints

TEACHER="${TEACHER:-multisenge_utae/checkpoints/run_c10_s2_gelu_full_v0/best.pt}"
OUT="${OUT:-multisenge_utae/checkpoints/cmu_s1_convnext_gelu_v0}"
RESUME="${RESUME:-$OUT/last.pt}"
if [[ ! -f "$TEACHER" ]]; then
  echo "missing teacher ckpt: $TEACHER"
  exit 1
fi
if [[ ! -f "$RESUME" ]]; then
  echo "missing resume ckpt: $RESUME"
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
