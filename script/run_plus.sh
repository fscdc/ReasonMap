#!/bin/bash
# ReasonMap-Plus evaluation with the transformers backend.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

model_list=(
    "Qwen/Qwen2.5-VL-7B-Instruct"
    "Qwen/Qwen2.5-VL-72B-Instruct"
    "moonshotai/Kimi-VL-A3B-Instruct"
    "moonshotai/Kimi-VL-A3B-Thinking"
    "Qwen/Qwen2.5-VL-32B-Instruct"
)

mkdir -p logs
for model in "${model_list[@]}"; do
    echo "Running model: $model"
    tag=$(basename "$model")
    python infer.py --task plus --backend hf --model "$model" \
        ${CACHE_DIR:+--cache-dir "$CACHE_DIR"} 2>&1 | tee "./logs/${model//\//-}_plus.txt"
    python score.py "outputs/plus/${tag}.jsonl"
done
