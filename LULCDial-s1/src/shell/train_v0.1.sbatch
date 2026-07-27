#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 08:00:00
#SBATCH -J ft_v01
#SBATCH -o /home/rihak_iitp/ft_v01_%j.out
#SBATCH -e /home/rihak_iitp/ft_v01_%j.err
#
# Fine-tune MultiSenGE 70/30 → LULCDial_S1_v0.1
# Do NOT force CUDA_VISIBLE_DEVICES=0 — let Slurm assign the free GPU.
# sbatch ~/MTP/earth2/LULCDial-s1/src/shell/train_v0.1.sbatch

module purge
module load MLDL/Pytorch-gpu
cd /home/rihak_iitp/MTP/earth2/LULCDial-s1/src
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
unset PYTORCH_CUDA_ALLOC_CONF

echo "=== CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>} ==="
nvidia-smi || true

python -m torch.distributed.run \
  --nnodes=1 --nproc_per_node=1 --standalone --master_port=34254 \
  earthdial/train/finetune.py \
  --model_name_or_path /home/rihak_iitp/EarthDial_Models/EarthDial_4B_MS \
  --conv_style "phi3-chat" \
  --output_dir /home/rihak_iitp/MTP/earth2/LULCDial-s1/checkpoints/LULCDial_S1_v0.1 \
  --meta_path shell/data/Stage4_BareSoil_S1_v0.1.json \
  --overwrite_output_dir True \
  --force_image_size 448 --bf16 True --num_train_epochs 1 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 128 \
  --grad_checkpoint True --freeze_backbone True --max_seq_length 1024 \
  --evaluation_strategy "no" --save_strategy "epoch" --logging_steps 1 \
  --dataloader_num_workers 0 \
  --learning_rate 4e-5 --do_train True
