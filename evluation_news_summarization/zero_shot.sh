#!/bin/bash

source ~/.bashrc
conda activate lf
cd ~/AMEFT/evluation_news_summarization

# MODEL_PATH="/data/models/huggingface/meta-llama/Llama-2-7b-hf"
# MODEL_PATH="mistralai/Mistral-7B-Instruct-v0.2"
MODEL_PATH="meta-llama/Meta-Llama-3-8B-Instruct"
# LORA_PATH="/data/user_data/wenkail/mlsys/lora_test_v3/llama_2_7b/checkpoint-100"
LORA_PATH="/data/user_data/wenkail/mlsys/lora_test_v4/llama_3_8b_instruct/checkpoint-400"
# LORA_PATH="/data/user_data/wenkail/mlsys/lora_test_v3/mistral_7b_instruct_v0.2"
DATASET_PATH="alpaca_news_summarization_test.json"
MODEL_NAME="new_llama_3_8b_with_lora_zero_shot"

CUDA_VISIBLE_DEVICES=1 \
    # python zero_shot.py \
    python zero_shot_with_lora.py \
    --model_name=$MODEL_NAME \
    --model_path=$MODEL_PATH \
    --dataset_path=$DATASET_PATH \
    --lora_path=$LORA_PATH
    # --use_lora True
    
