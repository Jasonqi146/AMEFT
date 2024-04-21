#!/bin/bash

MODEL_PATH="/data/models/huggingface/meta-llama/Llama-2-7b-hf"
OUTPUT_DIR="/data/user_data/wenkail/mlsys/lora_test"


source ~/.bashrc
cd ~/AMEFT/LLaMA-Factory/src
conda activate llama_factory

CUDA_VISIBLE_DEVICES=0 python train_bash.py \
    --stage sft \
    --do_train \
    --model_name_or_path=$MODEL_PATH \
    --dataset alpaca_news_summarization_train \
    --template default \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir=$OUTPUT_DIR \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 2048 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --warmup_steps 20 \
    --save_steps 100 \
    --eval_steps 100 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --max_samples 3000 \
    --val_size 0.1 \
    --plot_loss \
    --fp16