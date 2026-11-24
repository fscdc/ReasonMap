#!/bin/bash
# ReasonMap evaluation of open-source models with the transformers backend.
# Set CACHE_DIR to reuse an existing HF cache for datasets / models.

####### Reasoning models ########
# "Qwen/QVQ-72B-Preview"
# "Skywork/Skywork-R1V-38B"
# "Skywork/Skywork-R1V2-38B"
# "moonshotai/Kimi-VL-A3B-Thinking"

######## Non-reasoning models ########
# "Qwen/Qwen2.5-VL-3B-Instruct" "Qwen/Qwen2.5-VL-32B-Instruct" "Qwen/Qwen2.5-VL-72B-Instruct"
# "OpenGVLab/InternVL3-78B" "OpenGVLab/InternVL3-38B"

# A "-onlytext" suffix runs the text-only ablation (no map given to the model).
model_list=(
  "Qwen/QVQ-72B-Preview"
  "Qwen/Qwen2.5-VL-3B-Instruct"
  "Qwen/Qwen2.5-VL-3B-Instruct-onlytext"
  "Qwen/Qwen2.5-VL-32B-Instruct"
  "Qwen/Qwen2.5-VL-72B-Instruct"
  "Qwen/Qwen2.5-VL-72B-Instruct-onlytext"
  "OpenGVLab/InternVL3-38B"
  "OpenGVLab/InternVL3-78B"
  "Skywork/Skywork-R1V-38B"
  "moonshotai/Kimi-VL-A3B-Instruct"
  "moonshotai/Kimi-VL-A3B-Instruct-onlytext"
  "moonshotai/Kimi-VL-A3B-Thinking"
  "moonshotai/Kimi-VL-A3B-Thinking-onlytext"
)

mkdir -p logs
for model in "${model_list[@]}"; do
    echo "Running model: $model"
    tag=$(basename "$model")
    python infer.py --task reasonmap --backend hf --model "$model" \
        ${CACHE_DIR:+--cache-dir "$CACHE_DIR"} 2>&1 | tee "./logs/${model//\//-}.txt"
    python score.py "outputs/reasonmap/${tag}.jsonl"
done
