####### Reasoning models ########
# "Qwen/QVQ-72B-Preview"
# "Skywork/Skywork-R1V-38B"
# "Skywork/Skywork-R1V2-38B"
# "moonshotai/Kimi-VL-A3B-Thinking"

######## Non-reasoning models ########
# "Qwen/Qwen2.5-VL-3B-Instruct" "Qwen/Qwen2.5-VL-32B-Instruct" "Qwen/Qwen2.5-VL-72B-Instruct"
# "OpenGVLab/InternVL3-78B" "OpenGVLab/InternVL3-38B"

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

for model in "${model_list[@]}"; do
    echo "Running model: $model"
    log_name=${model//\//-}.txt
    python main.py --model "$model" \
        --cache_path "to-add-your-cache-dir" > ./logs/"$log_name"
done