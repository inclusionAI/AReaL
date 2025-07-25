#!/usr/bin/env bash

set -e

GIT_COMMIT_SHA=${GIT_COMMIT_SHA:?"GIT_COMMIT_SHA is not set"}

echo "GIT_COMMIT_SHA: $GIT_COMMIT_SHA"

RUN_ID="areal-$GIT_COMMIT_SHA"

# If there is already an image for the current environment, skip the build.
ENV_SHA=$(sha256sum pyproject.toml | awk '{print $1}')
if docker images --format '{{.Repository}}:{{.Tag}}' | grep -q "areal-env:$ENV_SHA"; then
    echo "Image areal-env already exists, skipping build."
    exit 0
fi

if docker ps -a --format '{{.Names}}' | grep -q "$RUN_ID"; then
    docker rm -f $RUN_ID
fi

docker run \
    --name $RUN_ID \
    --gpus all \
    --shm-size=8g \
    -v $(pwd):/workspace \
    -w /workspace \
    nvcr.io/nvidia/pytorch:25.01-py3 \
    bash -c "
        pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
        pip config unset global.extra-index-url
        pip uninstall -y --ignore-installed transformer-engine
        pip uninstall -y --ignore-installed torch-tensorrt
        pip uninstall -y --ignore-installed nvidia-dali-cuda120
        bash examples/env/scripts/setup-pip-deps.sh
    " || { docker rm -f $RUN_ID; exit 1; }

docker commit $RUN_ID "areal-env:$ENV_SHA"
docker rm -f $RUN_ID
