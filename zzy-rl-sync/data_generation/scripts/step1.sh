#!/usr/bin/env bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

SCRIPT_DIR="."
ROOT_DIR="."

OPENAI_KEY_PATH=${OPENAI_KEY_PATH:-~/.openai_api_key}
if [ -f "$OPENAI_KEY_PATH" ]; then
  export OPENAI_API_KEY=$(cat "$OPENAI_KEY_PATH")
elif [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "OPENAI_API_KEY is not set and OPENAI_KEY_PATH does not exist at $OPENAI_KEY_PATH" >&2
  exit 1
fi

VERSION=${VERSION:-v1}
NUM_SAMPLES=${NUM_SAMPLES:-20}
SRC_NAME=${SRC_NAME:-Qwen3_8B_bfloat16_fixed_qwen_template_40k_polaris_data_53K_1_1k}
INPUT_SUFFIX=${INPUT_SUFFIX:-}
DIR_NAME=${DIR_NAME:-${SRC_NAME}${INPUT_SUFFIX}_${NUM_SAMPLES}samples}
DATA_PATH=${ROOT_DIR}/data/${DIR_NAME}

python "$ROOT_DIR/src/gpt.py" \
    --prompt "$ROOT_DIR/prompt/step1-prompt_${VERSION}.txt" \
    --input "${DATA_PATH}/collected.jsonl" \
    --output "${DATA_PATH}_step1_${VERSION}/" \
    --chat "${DATA_PATH}_step1_${VERSION}/chat" \
    --openai_model gpt-5-2025-08-07 \
    --workers 128 \
    --skip_existing
