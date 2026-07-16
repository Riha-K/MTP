#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 08:00:00
#SBATCH -J ft_v02
#SBATCH -o /home/rihak_iitp/ft_v02_%j.out
#SBATCH -e /home/rihak_iitp/ft_v02_%j.err
#
# Fine-tune on MultiSenGE 70/30 train shards → LULCDial_S1_v0.2
# sbatch ~/MTP/earth2/LULCDial-s1/src/shell/train_v0.2.sbatch

module purge
module load MLDL/Pytorch-gpu
cd /home/rihak_iitp/MTP/earth2/LULCDial-s1/src
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m torch.distributed.run \
  --nnodes=1 --nproc_per_node=1 --master_port=34252 \
  earthdial/train/finetune.py \
  --model_name_or_path /home/rihak_iitp/EarthDial_Models/EarthDial_4B_MS \
  --conv_style "phi3-chat" \
  --output_dir /home/rihak_iitp/MTP/earth2/LULCDial-s1/checkpoints/LULCDial_S1_v0.2 \
  --meta_path shell/data/Stage4_BareSoil_S1_v0.2.json \
  --overwrite_output_dir True \
  --force_image_size 448 --bf16 True --num_train_epochs 1 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 128 \
  --grad_checkpoint True --freeze_backbone True --max_seq_length 1024 \
  --evaluation_strategy "no" --save_strategy "epoch" --logging_steps 1 \
  --learning_rate 4e-5 --do_train True
