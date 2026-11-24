#!/bin/bash
# Serve a model with SGLang, then run ReasonMap + ReasonMap-Plus inference against it and score.
#
#   bash script/run_sglang.sh Qwen/Qwen2.5-VL-7B-Instruct          # 1 GPU
#   TP=4 bash script/run_sglang.sh Qwen/Qwen2.5-VL-72B-Instruct    # tensor parallel
#
# Env: TP (default 1), PORT (30000), NUM_WORKERS (16), TASKS ("reasonmap plus"),
#      CONTEXT_LENGTH (32768), SGLANG_EXTRA_ARGS (passed to sglang.launch_server).
set -euo pipefail

MODEL=${1:?usage: run_sglang.sh <model> }
TP=${TP:-1}
PORT=${PORT:-30000}
NUM_WORKERS=${NUM_WORKERS:-16}
TASKS=${TASKS:-"reasonmap plus"}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-32768}
TAG=$(basename "$MODEL")
mkdir -p logs

python -m sglang.launch_server \
    --model-path "$MODEL" \
    --port "$PORT" \
    --tp "$TP" \
    --context-length "$CONTEXT_LENGTH" \
    --trust-remote-code \
    ${SGLANG_EXTRA_ARGS:-} > "logs/sglang_${TAG}.log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true; wait $SERVER_PID 2>/dev/null || true' EXIT

echo "Waiting for SGLang on port $PORT (log: logs/sglang_${TAG}.log) ..."
until curl -sf "http://localhost:${PORT}/v1/models" > /dev/null; do
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "SGLang exited; see logs/sglang_${TAG}.log"; exit 1; }
    sleep 10
done

for TASK in $TASKS; do
    python infer.py --task "$TASK" --backend sglang --base-url "http://localhost:${PORT}/v1" \
        --model "$MODEL" --num-workers "$NUM_WORKERS"
    python score.py "outputs/${TASK}/${TAG}.jsonl"
done
