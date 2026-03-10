#!/usr/bin/env bash
set -euo pipefail
model_path="/storage/openpsi/models/GLM-4.1V-9B-Thinking"
echo "model: $model_path"
export VLLM_ENGINE_ITERATION_TIMEOUT_S=600
mkdir -p /tmp/log/
nohup python -m vllm.entrypoints.openai.api_server \
  --model "$model_path" \
  --host 0.0.0.0 \
  --trust-remote-code \
  --port 8000 \
  --data-parallel-size 8 \
  --tensor-parallel-size 1 \
  --max-model-len 65536 \
  --dtype auto \
  --gpu-memory-utilization 0.8 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --enable-prefix-caching \
  > /tmp/log/vllm_api.log 2>&1 &
#  --chat-template-content-format string \
#chat-template-content-format string for internvl
echo "waiting endpoint..."
until curl -sf http://127.0.0.1:8000/v1/models > /dev/null; do
  tail -n 10  /tmp/log/vllm_api.log
  sleep 2
done
echo "endpoint ready"