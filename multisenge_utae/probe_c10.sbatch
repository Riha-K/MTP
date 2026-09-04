#!/bin/bash
# PARAM P3 10-class linear probes L0-L3 (val 31UFP+31UGP).
# Preferred (same as 6-class): after head best.pt exists
#   sbatch --exclude=ragpu004,ragpu005,ragpu007 multisenge_utae/probe_c10.sbatch
# Start now before head (random encoder — less sir-comparable):
#   ALLOW_RANDOM=1 sbatch --exclude=ragpu004,ragpu005,ragpu007 multisenge_utae/probe_c10.sbatch
#SBATCH -J utae_c10_probe
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 12:00:00
#SBATCH -o /home/rihak_iitp/MTP/earth2/multisenge_utae/artifacts/slurm-c10-probe-%j.out
#SBATCH -e /home/rihak_iitp/MTP/earth2/multisenge_utae/artifacts/slurm-c10-probe-%j.err

set -eo pipefail
cd /home/rihak_iitp/MTP/earth2
OUT="multisenge_utae/results/probe_c10_v0"
mkdir -p multisenge_utae/artifacts "${OUT}"

module purge
module load MLDL/Pytorch-gpu
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

CKPT="${CKPT:-multisenge_utae/checkpoints/run_c10_head_v0/best.pt}"
EXTRA=()

if [[ -f "${CKPT}" ]]; then
  echo "using ckpt ${CKPT}"
  EXTRA+=(--ckpt "${CKPT}")
elif [[ "${ALLOW_RANDOM:-0}" == "1" ]]; then
  echo "WARNING: no head ckpt — random-init encoder + estimated norm_stats"
else
  echo "ERROR: checkpoint not found: ${CKPT}"
  echo "Wait for train_c10_head.sbatch (job 100067), or set ALLOW_RANDOM=1"
  exit 1
fi

python -m multisenge_utae.probe_layers \
  --index multisenge_seg/artifacts/patch_index.json \
  --data-root LULCDial-s1/data/lulcdial_s1/ai4lcc/multisenge \
  --num-classes 10 \
  --probe linear \
  "${EXTRA[@]}" \
  --batch-size 2 \
  --workers 2 \
  --pixels-per-patch 512 \
  --out-dir "${OUT}"

echo "=== probe outputs ==="
ls -la "${OUT}"
