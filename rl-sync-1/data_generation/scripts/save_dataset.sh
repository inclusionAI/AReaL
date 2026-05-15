# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

export root_dir='.'
export VERSION=${VERSION:-v1_v1_v1}
export STEP5_NAME=${STEP5_NAME:-step5}
export NUM_SAMPLES=${NUM_SAMPLES:-1000}

export SUFFIX=${SUFFIX:-v111}
export SRC_NAME=${SRC_NAME:-polaris_data_53K_1_1k}
export DATA_PATH=${root_dir}/data/${SRC_NAME}${INPUT_SUFFIX}_${NUM_SAMPLES}samples
export DATA_SUFFIX=_${STEP5_NAME}_${VERSION}
export DATASET_PATH=${root_dir}/dataset/${SRC_NAME}${INPUT_SUFFIX}_${NUM_SAMPLES}samples_$SUFFIX

python src/save_dataset.py \
  --input  ${DATA_PATH}/collected.jsonl \
  --trajectory-glob "${DATA_PATH}${DATA_SUFFIX}/*.txt" \
  --output ${DATASET_PATH} \
  --trajectory-matched-only \
  --build-chat \
  --qwen-model Qwen/Qwen3-8B \
  --sample-size ${NUM_SAMPLES} \
  --repeat ${REPEAT:-8} \
  "$@"
