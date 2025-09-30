#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


model_list=(
    "Qwen/Qwen2.5-VL-7B-Instruct"
    "Qwen/Qwen2.5-VL-72B-Instruct"
    "moonshotai/Kimi-VL-A3B-Instruct"
    "moonshotai/Kimi-VL-A3B-Thinking"
    "Qwen/Qwen2.5-VL-32B-Instruct"
)

for model in "${model_list[@]}"; do
    echo "Running model: $model"
    log_name=${model//\//-}_plus.txt
    python main_plus.py --model_path "$model" \
        --cache_path "./cache" > ./logs/"$log_name"
done
