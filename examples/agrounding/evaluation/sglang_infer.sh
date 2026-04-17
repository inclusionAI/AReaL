#!/usr/bin/env bash
set -euo pipefail
model_path="${MODEL_PATH}"
data_json="${DATA_JSON}"
output_dir="${OUTPUT_DIR}"
prompt_template=qwen3
box_remap=scale

echo "model: $model_path"

mkdir -p ./log
nohup python -m sglang.launch_server \
  --model-path "$model_path" \
  --host 0.0.0.0 --port 30000 \
  --nnodes 1 --node-rank 0 \
  --dp-size 8 \
  --tp-size 1 \
  --dtype auto \
  --mem-fraction-static 0.8 \
  > ./log/sglang_node0.log 2>&1 &


echo "waiting endpoint..."
until curl -sf http://127.0.0.1:30000/v1/models > /dev/null; do
  tail -n 10  ./log/sglang_node0.log
  sleep 2
done
echo "endpoint ready"

python examples/agrounding/evaluation/sglang_infer.py \
    --model $model_path \
    --data_json $data_json \
    --output_dir $output_dir \
    --endpoint "http://127.0.0.1:30000" \
    --max_tokens 4096 \
    --concurrency 64 \
    --prompt_template $prompt_template \
    --box_remap $box_remap

pkill -9 "sglang" -f





