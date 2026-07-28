#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 05:00:00
#SBATCH -J zs_v01
#SBATCH -o /home/rihak_iitp/zs_v01_%j.out
#SBATCH -e /home/rihak_iitp/zs_v01_%j.err
#
# EarthDial ZS on 70/30 test (~2497).
# sbatch ~/MTP/earth2/LULCDial-s1/src/shell/pred_zs_v0.1.sbatch

module purge
module load MLDL/Pytorch-gpu
cd /home/rihak_iitp/MTP/earth2/LULCDial-s1
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

mkdir -p data/lulcdial_s1/bench/v0.1/preds/earthdial_zs

python -m lulcdial.predict_zero_shot \
  --bench-jsonl data/lulcdial_s1/bench/v0.1/ai4lcc_test.jsonl \
  --s1-root data/lulcdial_s1/ai4lcc/multisenge/s1_test_bench_v0.1 \
  --checkpoint /home/rihak_iitp/EarthDial_Models/EarthDial_4B_MS \
  --out-pred-jsonl data/lulcdial_s1/bench/v0.1/preds/earthdial_zs/ai4lcc_test_predictions.jsonl \
  --resume
