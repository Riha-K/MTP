#!/bin/bash
# 10-class S2-only ReLU teacher head. Set SEED (43-46). Does not touch seed-42 v0 folders.
#   SEED=43 sbatch --exclude=ragpu004,ragpu005,ragpu007 multisenge_utae/train_s2_c10_head_seed.sbatch
#SBATCH -J s2h_seed
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH -o /home/rihak_iitp/MTP/earth2/multisenge_utae/artifacts/slurm-s2-c10-head-seed-%j.out
#SBATCH -e /home/rihak_iitp/MTP/earth2/multisenge_utae/artifacts/slurm-s2-c10-head-seed-%j.err

set -eo pipefail
cd /home/rihak_iitp/MTP/earth2
if [[ -z "${SEED:-}" || "$SEED" == "42" ]]; then
  echo "set SEED to 43-46; seed 42 stays in run_c10_s2_head_v0"
  exit 1
fi
OUT="multisenge_utae/checkpoints/run_c10_s2_head_s${SEED}_v0"
mkdir -p multisenge_utae/artifacts "$OUT"
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
  --mode head \
  --modality s2 \
  --activation relu \
  --seed "$SEED" \
  --epochs 80 \
  --batch-size 2 \
  --workers 2 \
  --out-dir "$OUT"

python -m multisenge_utae.plot_history \
  "$OUT/history.json" \
  --title "U-TAE S2-only P4 head ReLU seed ${SEED} (10-class)" \
  || echo "plot skipped"
