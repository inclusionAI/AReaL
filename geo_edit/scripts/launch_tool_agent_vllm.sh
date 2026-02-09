#!/usr/bin/env bash
set -euo pipefail

tool_name=""
model_name_or_path=""
port=""
gpu_id=""
node_ip=""
host="0.0.0.0"
max_model_len="32768"
dtype="auto"
gpu_memory_utilization="0.9"
startup_timeout_s="300"
log_dir="/tmp/log/geo_edit_tool_agent"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tool-name)
      tool_name="$2"
      shift 2
      ;;
    --model-name-or-path)
      model_name_or_path="$2"
      shift 2
      ;;
    --port)
      port="$2"
      shift 2
      ;;
    --gpu-id)
      gpu_id="$2"
      shift 2
      ;;
    --node-ip)
      node_ip="$2"
      shift 2
      ;;
    --host)
      host="$2"
      shift 2
      ;;
    --max-model-len)
      max_model_len="$2"
      shift 2
      ;;
    --dtype)
      dtype="$2"
      shift 2
      ;;
    --gpu-memory-utilization)
      gpu_memory_utilization="$2"
      shift 2
      ;;
    --startup-timeout-s)
      startup_timeout_s="$2"
      shift 2
      ;;
    --log-dir)
      log_dir="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

if [[ -z "$tool_name" || -z "$model_name_or_path" || -z "$port" || -z "$gpu_id" ]]; then
  echo "Missing required args. Need --tool-name --model-name-or-path --port --gpu-id."
  exit 1
fi

mkdir -p "$log_dir"
log_file="${log_dir}/${tool_name}_${port}.log"
pid_file="${log_dir}/${tool_name}_${port}.pid"

export CUDA_VISIBLE_DEVICES="$gpu_id"
export VLLM_ENGINE_ITERATION_TIMEOUT_S=600

nohup python -m vllm.entrypoints.openai.api_server \
  --model "$model_name_or_path" \
  --host "$host" \
  --port "$port" \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --max-model-len "$max_model_len" \
  --dtype "$dtype" \
  --gpu-memory-utilization "$gpu_memory_utilization" \
  --enable-prefix-caching \
  >"$log_file" 2>&1 &
echo $! >"$pid_file"

deadline=$((SECONDS + startup_timeout_s))
health_host="127.0.0.1"
if [[ -n "$node_ip" ]]; then
  health_host="$node_ip"
elif [[ "$host" != "0.0.0.0" ]]; then
  health_host="$host"
fi
until curl -sf "http://${health_host}:${port}/v1/models" >/dev/null; do
  if (( SECONDS >= deadline )); then
    echo "Tool agent ${tool_name} failed to start in ${startup_timeout_s}s"
    tail -n 100 "$log_file" || true
    exit 1
  fi
  sleep 2
done

echo "Tool agent ${tool_name} ready at ${health_host}:${port}"
