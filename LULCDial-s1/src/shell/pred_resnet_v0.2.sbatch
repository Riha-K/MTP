#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 02:00:00
#SBATCH -J pred_resnet
#SBATCH -o /home/rihak_iitp/pred_resnet_%j.out
#SBATCH -e /home/rihak_iitp/pred_resnet_%j.err
#
# Predict ResNet-18 on v0.2 test bench (~2497) then score with eval_zero_shot.
# sbatch ~/MTP/earth2/LULCDial-s1/src/shell/pred_resnet_v0.2.sbatch

module purge
module load MLDL/Pytorch-gpu
cd /home/rihak_iitp/MTP/earth2/LULCDial-s1
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "=== CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>} ==="
nvidia-smi || true

mkdir -p data/baresoil_s1/bench/v0.2/preds/resnet18
mkdir -p data/baresoil_s1/metrics/v0.2

python -m baresoil.cnn.predict \
  --checkpoint checkpoints/resnet18_s1_v0.2/best.pt \
  --bench-jsonl data/baresoil_s1/bench/v0.2/ai4lcc_test.jsonl \
  --s1-root data/baresoil_s1/ai4lcc/multisenge/s1_test_bench_v0.2 \
  --out-pred-jsonl data/baresoil_s1/bench/v0.2/preds/resnet18/ai4lcc_test_predictions.jsonl \
  --arch resnet18 \
  --batch-size 64 \
  --threshold 0.5

python -m baresoil.eval_zero_shot \
  --bench-jsonl data/baresoil_s1/bench/v0.2/ai4lcc_test.jsonl \
  --pred-jsonl data/baresoil_s1/bench/v0.2/preds/resnet18/ai4lcc_test_predictions.jsonl \
  --out-metrics data/baresoil_s1/metrics/v0.2/resnet18_s1.json
