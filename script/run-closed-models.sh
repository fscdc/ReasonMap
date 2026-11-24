#!/bin/bash
# ReasonMap evaluation of the closed models from the paper (see reasonmap/backends/__init__.py: PRESETS).
# API keys are read from the environment:
#   GEMINI_API_KEY (Gemini), ARK_API_KEY (Doubao, Volcengine Ark), OPENAI_API_KEY (o3 / 4o)

model_list=(
  "Gemini"
  "Doubao-115"
  "Doubao-115-onlytext"
  "Doubao-415"
  "Doubao-415-onlytext"
  "Doubao-428"
  "o3"
  "4o"
)

mkdir -p logs
for model in "${model_list[@]}"; do
    echo "Running model: $model"
    python infer.py --task reasonmap --preset "$model" --num-workers "${NUM_WORKERS:-4}" 2>&1 | tee "./logs/${model}.txt"
    python score.py "outputs/reasonmap/${model}.jsonl"
done
