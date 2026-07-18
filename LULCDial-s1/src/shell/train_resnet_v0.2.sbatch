#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 04:00:00
#SBATCH -J resnet_v02
#SBATCH -o /home/rihak_iitp/resnet_v02_%j.out
#SBATCH -e /home/rihak_iitp/resnet_v02_%j.err
#
# Train ResNet-18 multi-label on MultiSenGE S1 (70/30).
# sbatch ~/MTP/earth2/LULCDial-s1/src/shell/train_resnet_v0.2.sbatch
# If GPU0 is busy on a shared node: export CUDA_VISIBLE_DEVICES=1 before python.

module purge
module load MLDL/Pytorch-gpu
cd /home/rihak_iitp/MTP/earth2/LULCDial-s1
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "=== CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>} ==="
nvidia-smi || true

python -m baresoil.cnn.train \
  --labels-dir data/baresoil_s1/ai4lcc/multisenge/labels \
  --s1-dir data/baresoil_s1/ai4lcc/multisenge/s1 \
  --out-dir checkpoints/resnet18_s1_v0.2 \
  --arch resnet18 \
  --epochs 20 \
  --batch-size 64 \
  --lr 1e-3 \
  --num-workers 4
