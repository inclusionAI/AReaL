#!/usr/bin/env bash
set -euo pipefail

# 1. 参数处理
raw_model_path="${1:-}"
if [ -z "$raw_model_path" ]; then
    echo "Usage: $0 <model_path> [split]"
    exit 1
fi
model_path="${raw_model_path%/}"

model_name="$(basename "$model_path")"
echo "Model: $model_name ($model_path)"

# 2. 准备日志目录
mkdir -p /tmp/log
log_file="/tmp/log/vllm_${model_name}.log"
# 先清空或创建日志文件，确保 tail 不会报错
: > "$log_file"

# 3. 启动 vLLM
# 建议加上 PYTHONUNBUFFERED=1 以便日志能实时输出，不被缓存
echo "Starting vLLM server..."
export PYTHONUNBUFFERED=1 

nohup python -m vllm.entrypoints.openai.api_server \
  --model "$model_path" \
  --served-model-name "$model_name" \
  --host 0.0.0.0 \
  --trust-remote-code \
  --port 8000 \
  --data-parallel-size 4 \
  --tensor-parallel-size 2 \
  --dtype auto \
  --gpu-memory-utilization 0.8 \
  > "$log_file" 2>&1 &

# 获取 vLLM PID
VLLM_PID=$!
echo "vLLM server PID: $VLLM_PID"

# 4. 启动 tail -f 实时输出日志到屏幕，并在后台运行
echo "Streaming logs to console (will stop when server is ready)..."
tail -f "$log_file" &
TAIL_PID=$!

# 5. 定义清理函数
cleanup() {
    # 如果脚本退出，先杀掉 tail 进程，防止日志乱窜
    if [ -n "${TAIL_PID:-}" ] && kill -0 $TAIL_PID 2>/dev/null; then
        kill $TAIL_PID
    fi
    
    echo ""
    echo "Stopping vLLM server (PID: $VLLM_PID)..."
    if kill -0 $VLLM_PID 2>/dev/null; then
        kill $VLLM_PID
        wait $VLLM_PID 2>/dev/null || true 
    fi
    echo "Cleaned up."
}
trap cleanup EXIT SIGINT SIGTERM

# 6. 等待 Endpoint 就绪
start_time=$(date +%s)
timeout=1200 

while true; do
    # 检查 vLLM 进程是否还活着
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        # 杀掉 tail 防止它一直卡着
        kill $TAIL_PID 2>/dev/null || true
        echo ""
        echo "Error: vLLM process died unexpectedly."
        exit 1
    fi

    # 尝试连接
    if curl -sf "http://127.0.0.1:8000/v1/models" > /dev/null; then
        # === 服务已就绪 ===
        # 杀掉 tail 进程，停止在屏幕上输出日志
        kill $TAIL_PID 2>/dev/null || true
        wait $TAIL_PID 2>/dev/null || true # 防止打印 Terminated 信息
        echo ""
        echo "Endpoint ready! Stopping log stream."
        break
    fi

    # 检查超时
    current_time=$(date +%s)
    elapsed=$((current_time - start_time))
    if [ $elapsed -gt $timeout ]; then
        kill $TAIL_PID 2>/dev/null || true
        echo ""
        echo "Error: Timeout waiting for vLLM."
        exit 1
    fi

    sleep 2
done

# 7. 准备输出目录
out_dir="outputs/${model_name}"
mkdir -p "$out_dir"
pred_file="${out_dir}/visulogic_preds.jsonl"
score_file="${out_dir}/visulogic_score.json"

# 8. 运行评测脚本
python examples/evaluation/visulogic/visulogic.py \
  --dataset "/storage/openpsi/data/VisuLogic/data/test-00000-of-00001.parquet" \
  --model "$model_name" \
  --base-url "http://127.0.0.1:8000/v1" \
  --concurrency 64 \
  --output "$pred_file" \
  --score-json "$score_file"

echo "Evaluation finished. Results saved to $out_dir"