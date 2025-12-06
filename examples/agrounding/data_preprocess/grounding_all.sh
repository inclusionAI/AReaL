#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR=${OUTPUT_DIR}
TRAIN_JSON=${TRAIN_JSON}
    
python examples/agrounding/data_preprocess/grounding_all.py \
    --input_json $TRAIN_JSON \
    --output_dir $OUTPUT_DIR \
    --format qwen \
    --num_workers 64
