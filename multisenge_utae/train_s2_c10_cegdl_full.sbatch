#!/bin/bash
# Phase 5 Composite B full: weighted CE + generalized Dice. Starts from cegdl head best.pt.
#   sbatch --dependency=afterok:111845 --exclude=ragpu004,ragpu005,ragpu007 multisenge_utae/train_s2_c10_cegdl_full.sbatch
#SBATCH -J s2f_cegdl
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH -o /home/rihak_iitp/MTP/earth2/multisenge_utae/artifacts/slurm-s2-c10-cegdl-full-%j.out
#SBATCH -e /home/rihak_iitp/MTP/earth2/multisenge_utae/artifacts/slurm-s2-c10-cegdl-full-%j.err

set -eo pipefail
cd /home/rihak_iitp/MTP/earth2
OUT="multisenge_utae/checkpoints/run_c10_s2_cegdl_full_v0"
HEAD="${HEAD:-multisenge_utae/checkpoints/run_c10_s2_cegdl_head_v0/best.pt}"
mkdir -p multisenge_utae/artifacts "$OUT"
if [[ -f "$OUT/last.pt" ]]; then
  echo "refusing to overwrite $OUT/last.pt"
  exit 1
fi
if [[ ! -f "$HEAD" ]]; then
  echo "missing head ckpt: $HEAD"
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
  --activation relu \
  --loss ce_gdl \
  --dice-lambda 1 \
  --seed 42 \
  --init-ckpt "$HEAD" \
  --epochs 80 \
  --batch-size 2 \
  --workers 2 \
  --out-dir "$OUT"

python -m multisenge_utae.plot_history \
  "$OUT/history.json" \
  --title "U-TAE S2-only full CE+GDL (10-class)" \
  || echo "plot skipped"
