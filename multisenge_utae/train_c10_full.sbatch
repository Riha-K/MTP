#!/bin/bash
# PARAM: P5 full fine-tune 10-class (after run_c10_head_v0/best.pt exists)
#   sbatch --exclude=ragpu004,ragpu005,ragpu007 multisenge_utae/train_c10_full.sbatch
#SBATCH -J utae_c10_full
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH -o /home/rihak_iitp/MTP/earth2/multisenge_utae/artifacts/slurm-c10-full-%j.out
#SBATCH -e /home/rihak_iitp/MTP/earth2/multisenge_utae/artifacts/slurm-c10-full-%j.err

set -eo pipefail
cd /home/rihak_iitp/MTP/earth2
mkdir -p multisenge_utae/artifacts multisenge_utae/checkpoints

module purge
module load MLDL/Pytorch-gpu
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

HEAD_CKPT="${HEAD_CKPT:-multisenge_utae/checkpoints/run_c10_head_v0/best.pt}"
OUT="multisenge_utae/checkpoints/run_c10_full_v0"

if [[ ! -f "${HEAD_CKPT}" ]]; then
  echo "ERROR: 10-class head ckpt missing: ${HEAD_CKPT}"
  echo "Run train_c10_head.sbatch (P4) first."
  exit 1
fi

python -m multisenge_utae.train \
  --index multisenge_seg/artifacts/patch_index.json \
  --data-root LULCDial-s1/data/lulcdial_s1/ai4lcc/multisenge \
  --num-classes 10 \
  --mode full \
  --init-ckpt "${HEAD_CKPT}" \
  --epochs 80 \
  --batch-size 2 \
  --workers 2 \
  --out-dir "${OUT}"

python -m multisenge_utae.plot_history \
  "${OUT}/history.json" \
  --title "U-TAE P5 full fine-tune (10-class)" \
  --out "${OUT}/history_plot.png" \
  || echo "plot skipped (install matplotlib if needed)"
