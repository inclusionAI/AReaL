#!/usr/bin/env bash
set -euo pipefail
model_path="${1:-/storage/openpsi/models/lcy_image_edit/sft_workspace/qwen3vl8b-thinking-chartqa-1third}"
port="${2:-8000}"
echo "model: $model_path"
echo "port: $port"
export VLLM_ENGINE_ITERATION_TIMEOUT_S=600
mkdir -p /tmp/log/
nohup python -m vllm.entrypoints.openai.api_server \
  --model "$model_path" \
  --host 0.0.0.0 \
  --trust-remote-code \
  --port "$port" \
  --data-parallel-size "${DP_SIZE:-8}" \
  --tensor-parallel-size "${TP_SIZE:-1}" \
  --max-model-len "${MAX_MODEL_LEN:-65536}" \
  --dtype auto \
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.8}" \
  --enable-prefix-caching \
  > /tmp/log/vllm_api.log 2>&1 &
echo $! > /tmp/log/vllm.pid
echo "waiting endpoint..."
until curl -sf "http://127.0.0.1:${port}/v1/models" > /dev/null; do
  tail -n 10  /tmp/log/vllm_api.log
  sleep 2
done
echo "endpoint ready"