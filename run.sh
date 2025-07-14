#!/usr/bin/env bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate /research/projects/trans_llm/Yanshu_Li/eval/env_eval
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m accelerate.commands.launch --num_processes 4 train_dpo.py \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --local_dataset_path  \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-6 \
  --max_steps -1 \
  --beta 0.1 \
  --deepspeed_config ds_config.json \
  --output_dir  \
  --num_train_epochs 4 \
  --loss_type sigmoid \
 
