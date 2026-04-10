#!/usr/bin/env bash
set -euo pipefail
cd /storage/openpsi/users/lichangye.lcy/antoinegg1/AReaL

MODEL="/storage/openpsi/models/Qwen3-VL-8B-Instruct"
TAG="qwen3vl8b_instruct"
DATASET="/storage/openpsi/data/mm_mapqa/processed/mm_mapqa_flat.parquet"
OUTPUT="/storage/openpsi/data/mm_mapqa/results/${TAG}"
EVAL_OUTPUT="/storage/openpsi/data/mm_mapqa/results/eval_${TAG}"
PORT="${PORT:-8000}"
WORKERS="${WORKERS:-32}"

echo "=== [$(date)] Launching vLLM: Instruct ==="
bash geo_edit/scripts/launch_vllm_generate.sh "$MODEL" "$PORT"

echo "=== [$(date)] Starting inference: ${TAG} ==="
python -m geo_edit.scripts.direct_generate \
    --dataset_path "$DATASET" \
    --output_dir "$OUTPUT" \
    --dataset_name mm_mapqa \
    --model_name_or_path "$MODEL" \
    --api_base "http://localhost:${PORT}" \
    --model_type vLLM \
    --api_mode chat_completions \
    --max_concurrent_requests "$WORKERS"

echo "=== [$(date)] Inference done, evaluating ==="
python -m geo_edit.evaluation.eval_mm_mapqa \
    --result_path "$OUTPUT" \
    --output_path "$EVAL_OUTPUT"

echo "=== [$(date)] Stopping vLLM ==="
kill $(cat /tmp/log/vllm.pid) 2>/dev/null || true
rm -f /tmp/log/vllm.pid

echo ""
echo "========== Instruct Results =========="
cat "${EVAL_OUTPUT}/summary.txt"
echo ""
echo "Error IDs: ${EVAL_OUTPUT}/error_ids.json"
