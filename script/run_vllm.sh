#!/bin/bash
# Serve a model with vLLM, then run ReasonMap + ReasonMap-Plus inference against it and score.
#
#   bash script/run_vllm.sh Qwen/Qwen2.5-VL-7B-Instruct            # 1 GPU
#   TP=4 bash script/run_vllm.sh Qwen/Qwen2.5-VL-72B-Instruct      # tensor parallel
#
# Env: TP (tensor parallel size, default 1), PORT (8000), NUM_WORKERS (concurrent requests, 16),
#      TASKS ("reasonmap plus"), MAX_MODEL_LEN (32768), VLLM_EXTRA_ARGS (passed to `vllm serve`).
set -euo pipefail

MODEL=${1:?usage: run_vllm.sh <model> }
TP=${TP:-1}
PORT=${PORT:-8000}
NUM_WORKERS=${NUM_WORKERS:-16}
TASKS=${TASKS:-"reasonmap plus"}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
TAG=$(basename "$MODEL")
mkdir -p logs

vllm serve "$MODEL" \
    --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --max-model-len "$MAX_MODEL_LEN" \
    --limit-mm-per-prompt '{"image": 1}' \
    --trust-remote-code \
    ${VLLM_EXTRA_ARGS:-} > "logs/vllm_${TAG}.log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true; wait $SERVER_PID 2>/dev/null || true' EXIT

echo "Waiting for vLLM on port $PORT (log: logs/vllm_${TAG}.log) ..."
until curl -sf "http://localhost:${PORT}/v1/models" > /dev/null; do
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "vLLM exited; see logs/vllm_${TAG}.log"; exit 1; }
    sleep 10
done

for TASK in $TASKS; do
    python infer.py --task "$TASK" --backend vllm --base-url "http://localhost:${PORT}/v1" \
        --model "$MODEL" --num-workers "$NUM_WORKERS"
    python score.py "outputs/${TASK}/${TAG}.jsonl"
done
