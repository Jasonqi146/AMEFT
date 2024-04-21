#!/bin/bash

# MODEL_NAME_OR_PATH="/data/models/huggingface/meta-llama/Llama-2-7b-hf"
MODEL_NAME_OR_PATH="mistralai/Mistral-7B-Instruct-v0.2"
OUTPUT_DIR="/data/user_data/wenkail/mlsys/lora_test/mistral_7b_instruct_v0.2"


source ~/.bashrc
cd ~/AMEFT/LLaMA-Factory/src
conda activate llama_factory

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --config_file ../examples/accelerate/single_config.yaml \
    train_bash.py \
    --stage sft \
    --do_train \
    --model_name_or_path=$MODEL_NAME_OR_PATH \
    --dataset alpaca_news_summarization_train \
    --dataset_dir ../data \
    --template default \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir=$OUTPUT_DIR \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 2048 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 4 \
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
    --fp16
