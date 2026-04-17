#!/usr/bin/env bash
set -euo pipefail

VAL_DIR=${VAL_DIR}
OUTPUT_DIR=${OUTPUT_DIR}

VALID_FILES=(
  "refcoco_testA.jsonl"
  "refcoco_testB.jsonl"
  "refcoco+_testA.jsonl"
  "refcoco+_testB.jsonl"
  "refcoco+_val.jsonl"
  "refcocog_val.jsonl"
  "refcocog_test.jsonl"
  "refcoco_val.jsonl"
)

python examples/agrounding/data_preprocess/grounding_val.py \
  --input_dir "${VAL_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --test_files "${VALID_FILES[@]}" \
  --format qwen \
  --num_workers 64
