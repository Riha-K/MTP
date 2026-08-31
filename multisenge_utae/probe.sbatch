#!/bin/bash
#SBATCH -J msge_utae_probe
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 04:00:00
#SBATCH -o /home/rihak_iitp/MTP/earth2/multisenge_utae/artifacts/slurm-probe-%j.out
#SBATCH -e /home/rihak_iitp/MTP/earth2/multisenge_utae/artifacts/slurm-probe-%j.err

set -euo pipefail
cd /home/rihak_iitp/MTP/earth2
mkdir -p multisenge_utae/artifacts multisenge_utae/results

module purge
module load MLDL/Pytorch-gpu
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

# Optional: pass trained head ckpt for meaningful probes
CKPT="${CKPT:-}"

EXTRA=()
if [[ -n "${CKPT}" ]]; then
  EXTRA+=(--ckpt "${CKPT}")
fi

python -m multisenge_utae.probe_layers \
  --index multisenge_seg/artifacts/patch_index.json \
  --data-root LULCDial-s1/data/lulcdial_s1/ai4lcc/multisenge \
  --num-classes 6 \
  --probe linear \
  "${EXTRA[@]}" \
  --out-dir multisenge_utae/results/probe_c6_v0
