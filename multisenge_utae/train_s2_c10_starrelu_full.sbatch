#!/bin/bash
# 10-class S2-only U-TAE, P5 full, StarReLU, seed 42.
# Starts from run_c10_s2_starrelu_head_v0. Does not touch the ReLU, GELU, or SiLU checkpoints.
#   sbatch --dependency=afterok:HEAD_JOB --exclude=ragpu004,ragpu005,ragpu007 multisenge_utae/train_s2_c10_starrelu_full.sbatch
#SBATCH -J utae_s2_c10_srelu_f
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH -o /home/rihak_iitp/MTP/earth2/multisenge_utae/artifacts/slurm-s2-c10-starrelu-full-%j.out
#SBATCH -e /home/rihak_iitp/MTP/earth2/multisenge_utae/artifacts/slurm-s2-c10-starrelu-full-%j.err

set -eo pipefail
cd /home/rihak_iitp/MTP/earth2
mkdir -p multisenge_utae/artifacts multisenge_utae/checkpoints

HEAD="${HEAD:-multisenge_utae/checkpoints/run_c10_s2_starrelu_head_v0/best.pt}"
OUT=multisenge_utae/checkpoints/run_c10_s2_starrelu_full_v0
if [[ ! -f "$HEAD" ]]; then
  echo "missing head ckpt: $HEAD"
  exit 1
fi
if [[ -f "$OUT/last.pt" ]]; then
  echo "refusing to overwrite $OUT/last.pt"
  exit 1
fi

module purge
module load MLDL/Pytorch-gpu
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

python -m multisenge_utae.train \
  --index multisenge_seg/artifacts/patch_index.json \
  --data-root LULCDial-s1/data/lulcdial_s1/ai4lcc/multisenge \
  --num-classes 10 \
  --mode full \
  --modality s2 \
  --activation starrelu \
  --seed 42 \
  --init-ckpt "$HEAD" \
  --epochs 80 \
  --batch-size 2 \
  --workers 2 \
  --out-dir "$OUT"

python -m multisenge_utae.plot_history \
  "$OUT/history.json" \
  --title "U-TAE S2-only P5 full StarReLU (10-class)" \
  || echo "plot skipped"
