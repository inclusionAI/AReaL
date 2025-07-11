#!/usr/bin/env bash

set -e

GIT_COMMIT_SHA=${GIT_COMMIT_SHA:?"GIT_COMMIT_SHA is not set"}

echo "GIT_COMMIT_SHA: $GIT_COMMIT_SHA"

RUN_ID="areal-$GIT_COMMIT_SHA"
cd "/tmp/$RUN_ID"

if docker ps -a --format '{{.Names}}' | grep -q "$RUN_ID"; then
    docker rm -f $RUN_ID
fi

ENV_SHA=$(sha256sum pyproject.toml | awk '{print $1}')
docker run \
    --name $RUN_ID \
    --gpus all \
    --shm-size=8g \
    -v $(pwd):/workspace \
    -w /workspace \
    "areal-env:$ENV_SHA" \
    bash -c "
        mv /sglang ./sglang
        HF_ENDPOINT=https://hf-mirror.com python -m pytest -s arealite/
    " || { docker rm -f $RUN_ID; exit 1; }

docker rm -f $RUN_ID
