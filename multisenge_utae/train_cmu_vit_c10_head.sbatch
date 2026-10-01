#!/bin/bash
# Stage 2: CMU-ViT U-TAE 10c P4 head (CONCAT, S2 skips only).
# Same epoch-70 student as 6c P4. Does not wait for the 6c run.
#   sbatch --exclude=ragpu004,ragpu005,ragpu007 multisenge_utae/train_cmu_vit_c10_head.sbatch
#SBATCH -J cmuvit_c10_h
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH -o /home/rihak_iitp/MTP/earth2/multisenge_utae/logs/slurm-cmuvit-c10-head-%j.out
#SBATCH -e /home/rihak_iitp/MTP/earth2/multisenge_utae/logs/slurm-cmuvit-c10-head-%j.err

set -eo pipefail
cd /home/rihak_iitp/MTP/earth2
mkdir -p multisenge_utae/artifacts multisenge_utae/checkpoints multisenge_utae/logs

module purge
module load MLDL/Pytorch-gpu
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

STUDENT="${STUDENT:-multisenge_utae/checkpoints/cmu_s1_vit_v1/student_best.pt}"
S2="${S2:-multisenge_utae/checkpoints/run_c10_s2_full_v0/best.pt}"
if [[ ! -f "$STUDENT" ]]; then
  echo "missing Stage1 student: $STUDENT"
  exit 1
fi
if [[ ! -f "$S2" ]]; then
  echo "missing S2 teacher: $S2"
  exit 1
fi

python -m multisenge_utae.train_cmu_vit \
  --index multisenge_seg/artifacts/patch_index.json \
  --data-root LULCDial-s1/data/lulcdial_s1/ai4lcc/multisenge \
  --num-classes 10 \
  --mode head \
  --fusion concat \
  --student-ckpt "$STUDENT" \
  --s2-ckpt "$S2" \
  --epochs 80 \
  --batch-size 2 \
  --workers 2 \
  --patience 20 \
  --out-dir multisenge_utae/checkpoints/cmu_vit_c10_head_v0

python -m multisenge_utae.plot_history \
  multisenge_utae/checkpoints/cmu_vit_c10_head_v0/history.json \
  --title "CMU-ViT U-TAE P4 head (10-class)" \
  || echo "plot skipped"
