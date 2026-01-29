#!/usr/bin/env bash
set -euo pipefail

# ============================================
# vLLM Multi-Node Distributed Inference Script
# Supports DP (Data Parallel) + TP (Tensor Parallel)
# ============================================

# --- Configuration ---
model_path="/storage/openpsi/models/Qwen3-VL-235B-A22B-Thinking/"
head_node_ip="${HEAD_NODE_IP:-}"
node_role="${NODE_ROLE:-head}"  # "head" or "worker"
num_nodes="${NUM_NODES:-2}"
tensor_parallel_size="${TP_SIZE:-8}"
data_parallel_size="${DP_SIZE:-2}"
port=8000
ray_port=6379

echo "=========================================="
echo "vLLM Multi-Node Launch Configuration"
echo "=========================================="
echo "Model: $model_path"
echo "Node Role: $node_role"
echo "Num Nodes: $num_nodes"
echo "Tensor Parallel Size: $tensor_parallel_size"
echo "Data Parallel Size: $data_parallel_size"
echo "=========================================="

export VLLM_ENGINE_ITERATION_TIMEOUT_S=600
mkdir -p /tmp/log/

# --- Start Ray Cluster ---
if [ "$node_role" == "head" ]; then
    echo "Starting Ray head node..."
    ray stop --force 2>/dev/null || true
    ray start --head --port=$ray_port --num-gpus=8

    echo "Waiting for worker nodes to join..."
    sleep 10

    echo "Ray cluster status:"
    ray status

    echo "Starting vLLM API server on head node..."
    nohup python -m vllm.entrypoints.openai.api_server \
        --model "$model_path" \
        --host 0.0.0.0 \
        --trust-remote-code \
        --port $port \
        --tensor-parallel-size $tensor_parallel_size \
        --data-parallel-size $data_parallel_size \
        --max-model-len 65536 \
        --dtype auto \
        --gpu-memory-utilization 0.8 \
        --enable-auto-tool-choice \
        --tool-call-parser hermes \
        > /tmp/log/vllm_api.log 2>&1 &

    echo "Waiting for endpoint..."
    until curl -sf http://127.0.0.1:$port/v1/models > /dev/null; do
        tail -n 10 /tmp/log/vllm_api.log
        sleep 5
    done
    echo "Endpoint ready at http://0.0.0.0:$port"

elif [ "$node_role" == "worker" ]; then
    if [ -z "$head_node_ip" ]; then
        echo "Error: HEAD_NODE_IP must be set for worker nodes"
        exit 1
    fi

    echo "Starting Ray worker node, connecting to head at $head_node_ip:$ray_port..."
    ray stop --force 2>/dev/null || true
    ray start --address="$head_node_ip:$ray_port" --num-gpus=8

    echo "Worker node joined the cluster."
    ray status
    echo "Worker node is ready and waiting for tasks."

    # Keep the worker running
    tail -f /dev/null
else
    echo "Error: NODE_ROLE must be 'head' or 'worker'"
    exit 1
fi
