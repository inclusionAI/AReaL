#!/usr/bin/env bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

SCRIPT_DIR="."
ROOT_DIR="."

VERSION=${VERSION:-v1.1}
NUM_SAMPLES=${NUM_SAMPLES:-20}
SRC_NAME=${SRC_NAME:-Qwen3_8B_bfloat16_fixed_qwen_template_40k_polaris_data_53K_1_1k}
INPUT_SUFFIX=${INPUT_SUFFIX:-}
DIR_BASE_NAME=${BASE_NAME:-${SRC_NAME}${INPUT_SUFFIX}}
DIR_NAME=${DIR_NAME:-${SRC_NAME}${INPUT_SUFFIX}_${NUM_SAMPLES}samples}
DATA_PATH=${ROOT_DIR}/data/${DIR_NAME}
FILTER_DIFF_THRESHOLD=${FILTER_DIFF_THRESHOLD:-200}

INPUT_DIR=$DATA_PATH
STRUCT_DIR=${DATA_PATH}_step1_${VERSION}

if [ "$FILTER_DIFF_THRESHOLD" -eq 0 ]; then
    OUTPUT_DIR=${DATA_PATH}_step2_${VERSION}_filter_diff_th_0
else
    OUTPUT_DIR=${DATA_PATH}_step2_${VERSION}
fi

if [ -d "$OUTPUT_DIR" ]; then
    echo "Output directory already exists: $OUTPUT_DIR" >&2
    echo "Remove it or rerun with --overwrite passed to src/extract_v1.py." >&2
    exit 1
fi

success=0
while IFS= read -r txtfile; do
    base=$(basename "$txtfile" .txt)
    structfile="$STRUCT_DIR/${DIR_BASE_NAME}-${base}_reasoning.txt"
    echo "Running extract for $txtfile with $structfile"
    if python "$ROOT_DIR/src/extract_v1.py" -t "$txtfile" -a "$structfile" -o "$OUTPUT_DIR" --filter-diff-threshold "$FILTER_DIFF_THRESHOLD"; then
        success=1
    else
        echo "Extraction failed for $txtfile" >&2
    fi
done < <(find "$INPUT_DIR" -type f -name "*.txt" | sort)

if [ "$success" -ne 1 ]; then
    echo "No extraction succeeded." >&2
    exit 1
fi

exit 0
