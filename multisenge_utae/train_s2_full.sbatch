#!/bin/bash
# U-TAE S2-only 6c P5 full (after run_c6_s2_head_v0/best.pt).
#   sbatch --exclude=ragpu004,ragpu005,ragpu007 multisenge_utae/train_s2_full.sbatch
#SBATCH -J utae_s2_full
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH -o /home/rihak_iitp/MTP/earth2/multisenge_utae/artifacts/slurm-s2-full-%j.out
#SBATCH -e /home/rihak_iitp/MTP/earth2/multisenge_utae/artifacts/slurm-s2-full-%j.err

set -eo pipefail
cd /home/rihak_iitp/MTP/earth2
mkdir -p multisenge_utae/artifacts multisenge_utae/checkpoints

module purge
module load MLDL/Pytorch-gpu
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

HEAD="${HEAD:-multisenge_utae/checkpoints/run_c6_s2_head_v0/best.pt}"
if [[ ! -f "$HEAD" ]]; then
  echo "missing head ckpt: $HEAD"
  exit 1
fi

python -m multisenge_utae.train \
  --index multisenge_seg/artifacts/patch_index.json \
  --data-root LULCDial-s1/data/lulcdial_s1/ai4lcc/multisenge \
  --num-classes 6 \
  --mode full \
  --modality s2 \
  --init-ckpt "$HEAD" \
  --epochs 80 \
  --batch-size 2 \
  --workers 2 \
  --out-dir multisenge_utae/checkpoints/run_c6_s2_full_v0

python -m multisenge_utae.plot_history \
  multisenge_utae/checkpoints/run_c6_s2_full_v0/history.json \
  --title "U-TAE S2-only P5 full (6-class)" \
  || echo "plot skipped"
