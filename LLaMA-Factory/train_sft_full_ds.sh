#!/bin/bash
MODEL_NAME_OR_PATH="meta-llama/Meta-Llama-3-8B-Instruct"
OUTPUT_DIR="/workspace/AMEFT/LLaMA-Factory/full_tuning_output/meta-llama/Meta-Llama-3-8B-Instruct"

export HF_TOKEN=

deepspeed --num_gpus 1 \
    src/train_bash.py \
    --deepspeed examples/deepspeed/ds_z3_config.json \
    --stage sft \
    --do_train \
    --model_name_or_path=$MODEL_NAME_OR_PATH \
    --dataset alpaca_news_summarization_train \
    --dataset_dir ./data \
    --template default \
    --finetuning_type full \
    --output_dir=$OUTPUT_DIR \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 2048 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 1 \
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
    --num_train_epochs 10.0 \
    --max_samples 1000 \
    --val_size 0.1 \
    --plot_loss \
    --fp16 \
    --report_to 'wandb'
