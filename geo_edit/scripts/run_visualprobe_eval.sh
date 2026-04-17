#!/usr/bin/env bash
set -x

API_BASE=${API_BASE:-"http://127.0.0.1:8000"}
MODEL_PATH=${MODEL_PATH:-"/storage/openpsi/models/Qwen3-VL-8B-Thinking/"}
MODEL_TYPE=${MODEL_TYPE:-"vLLM"}
OUTPUT_BASE=${OUTPUT_BASE:-"/storage/openpsi/data/lcy_image_edit/visualprobe_test_0417"}
MAX_CONCURRENT=${MAX_CONCURRENT:-64}
SAMPLE_RATE=${SAMPLE_RATE:-1.0}

for level in easy medium hard; do
    LEVEL_UPPER=$(echo "$level" | sed 's/./\U&/')
    echo "========== VisualProbe ${LEVEL_UPPER} =========="
    python -m geo_edit.scripts.direct_generate \
        --api_base "$API_BASE" \
        --dataset_path "/storage/openpsi/data/VisualProbe_${LEVEL_UPPER}/val.parquet" \
        --dataset_name "visual_probe_${level}" \
        --output_dir "${OUTPUT_BASE}/${level}" \
        --model_name_or_path "$MODEL_PATH" \
        --model_type "$MODEL_TYPE" \
        --api_mode chat_completions \
        --max_concurrent_requests "$MAX_CONCURRENT" \
        --sample_rate "$SAMPLE_RATE"
done

echo "All VisualProbe evaluations finished."
