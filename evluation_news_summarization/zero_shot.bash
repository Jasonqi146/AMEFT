#!/bin/bash

source ~/.bashrc
conda activate llm_personality
cd ~/AMEFT/evluation_news_summarization

MODEL_PATH="/data/models/huggingface/meta-llama/Llama-2-7b-hf"
LORA_PATH="/data/user_data/wenkail/mlsys/lora_test_v2/llama_7b"
DATASET_PATH="alpaca_news_summarization_test.json"
MODEL_NAME="llama_7b"

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --config_file single_config.yaml \
    zero_shot.py \
    --model_name=$MODEL_NAME \
    --model_path=$MODEL_PATH \
    --dataset_path=$DATASET_PATH \
    --lora_path=#LORA_PATH
