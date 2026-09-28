#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${1:-${MODEL_PATH:-}}
if [ -z "$MODEL_PATH" ]; then
  echo "Usage: $0 /path/to/model-or-hf-repo"
  echo "Or set MODEL_PATH=/path/to/model-or-hf-repo"
  exit 2
fi

HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-trm-reward}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.88}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-32}
IMAGE_LIMIT=${IMAGE_LIMIT:-2}
VLLM_CMD=${VLLM_BIN:-vllm}
EXTRA_ARGS=${VLLM_EXTRA_ARGS:-}
unset VLLM_BIN VLLM_EXTRA_ARGS

exec "$VLLM_CMD" serve "$MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --trust-remote-code \
  --limit-mm-per-prompt "{\"image\": ${IMAGE_LIMIT}}" \
  $EXTRA_ARGS
