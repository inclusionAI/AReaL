#!/usr/bin/env bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

NUM_SAMPLES=${NUM_SAMPLES:-1000}
SRC_NAME=${SRC_NAME:-polaris_data_53K_1_1k}
OPENAI_KEY_PATH=${OPENAI_KEY_PATH:-~/.openai_api_key}
INPUT_SUFFIX=""
DATASET_BASE=${DATASET_BASE:-"data/processed/${SRC_NAME}"}
DATASET_PATH=${DATASET_PATH:-${DATASET_BASE}}

export INPUT_SUFFIX
export NUM_SAMPLES
export SRC_NAME
export OPENAI_KEY_PATH

DATA_DIR_NAME="${SRC_NAME}${INPUT_SUFFIX}_${NUM_SAMPLES}samples"
OUTPUT_PATH="data/${DATA_DIR_NAME}"

python src/collect_trajectories.py \
  --dataset_path "$DATASET_PATH" \
  --output_path "$OUTPUT_PATH" \
  --max_samples "$NUM_SAMPLES"

VERSION=v1 scripts/step1.sh
VERSION=v1 scripts/step2.sh
STEP2_VERSION=v1 EFFORT=low scripts/step3.sh
STEP3_VERSION=v1_v1 EFFORT=low scripts/step4.sh
VERSION=v1_v1_v1 scripts/step5.sh
VERSION=v1_v1_v1 SUFFIX=v111 scripts/save_dataset.sh
