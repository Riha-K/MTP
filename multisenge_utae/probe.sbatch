#!/bin/bash
# PARAM submit (from ~/MTP/earth2):
#   sbatch --exclude=ragpu004,ragpu005,ragpu007 multisenge_utae/probe.sbatch
#
# P3 layer probes L0-L3 on val tiles 31UFP+31UGP (train patches used to fit probe).
# Uses P4 head ckpt for norm_stats + weights (encoder was frozen in P4).
# Smoke first: sbatch multisenge_utae/probe_smoke.sbatch
#SBATCH -J msge_utae_probe
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 12:00:00
#SBATCH -o /home/rihak_iitp/MTP/earth2/multisenge_utae/artifacts/slurm-probe-%j.out
#SBATCH -e /home/rihak_iitp/MTP/earth2/multisenge_utae/artifacts/slurm-probe-%j.err

set -eo pipefail
cd /home/rihak_iitp/MTP/earth2
mkdir -p multisenge_utae/artifacts multisenge_utae/results/probe_c6_v0

module purge
module load MLDL/Pytorch-gpu
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

CKPT="${CKPT:-multisenge_utae/checkpoints/run_c6_head_v0/best.pt}"
OUT="multisenge_utae/results/probe_c6_v0"

if [[ ! -f "${CKPT}" ]]; then
  echo "ERROR: checkpoint not found: ${CKPT}"
  echo "Set CKPT=... or train P4 head first (train.sbatch)."
  exit 1
fi

python -m multisenge_utae.probe_layers \
  --index multisenge_seg/artifacts/patch_index.json \
  --data-root LULCDial-s1/data/lulcdial_s1/ai4lcc/multisenge \
  --num-classes 6 \
  --probe linear \
  --ckpt "${CKPT}" \
  --batch-size 2 \
  --workers 2 \
  --pixels-per-patch 512 \
  --out-dir "${OUT}"

echo "=== probe outputs ==="
ls -la "${OUT}"
