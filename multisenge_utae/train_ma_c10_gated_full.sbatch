#!/bin/bash
# PARAM: Task M — MA-UTAE gated 10c P5 full (after ma_c10_gated_head_v0/best.pt).
#   sbatch --exclude=ragpu004,ragpu005,ragpu007 multisenge_utae/train_ma_c10_gated_full.sbatch
# Then test:
#   CKPT=multisenge_utae/checkpoints/ma_c10_gated_full_v0/best.pt \
#   OUT=multisenge_utae/results/ma_utae/ma_c10_gated_full_v0 \
#   sbatch --exclude=ragpu004,ragpu005,ragpu007 multisenge_utae/eval_ma_c10.sbatch
#SBATCH -J ma_c10_g_full
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH -o /home/rihak_iitp/MTP/earth2/multisenge_utae/artifacts/slurm-ma-c10-g-full-%j.out
#SBATCH -e /home/rihak_iitp/MTP/earth2/multisenge_utae/artifacts/slurm-ma-c10-g-full-%j.err

set -eo pipefail
cd /home/rihak_iitp/MTP/earth2
mkdir -p multisenge_utae/artifacts multisenge_utae/checkpoints

module purge
module load MLDL/Pytorch-gpu
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

HEAD="${HEAD:-multisenge_utae/checkpoints/ma_c10_gated_head_v0/best.pt}"
if [[ ! -f "$HEAD" ]]; then
  echo "missing head ckpt: $HEAD"
  exit 1
fi

python -m multisenge_utae.train_ma \
  --index multisenge_seg/artifacts/patch_index.json \
  --data-root LULCDial-s1/data/lulcdial_s1/ai4lcc/multisenge \
  --num-classes 10 \
  --mode full \
  --fusion gated \
  --init-ckpt "$HEAD" \
  --epochs 80 \
  --batch-size 2 \
  --workers 2 \
  --out-dir multisenge_utae/checkpoints/ma_c10_gated_full_v0

python -m multisenge_utae.plot_history \
  multisenge_utae/checkpoints/ma_c10_gated_full_v0/history.json \
  --title "MA-UTAE gated P5 full (10-class)" \
  || echo "plot skipped"
