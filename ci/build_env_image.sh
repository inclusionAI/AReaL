#!/usr/bin/env bash

set -e

GIT_COMMIT_SHA=${GIT_COMMIT_SHA:?"GIT_COMMIT_SHA is not set"}

echo "GIT_COMMIT_SHA: $GIT_COMMIT_SHA"

# If there is already an image named areal-env, skip.
if docker images --format '{{.Repository}}:{{.Tag}}' | grep -q 'areal-env:latest'; then
    echo "Image areal-env already exists, skipping build."
    exit 0
fi

RUN_ID="areal-$GIT_COMMIT_SHA"
cd "/tmp/$RUN_ID"

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
        python -m pip install --upgrade pip
        pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
        pip config unset global.extra-index-url
        bash examples/env/scripts/setup-pip-deps.sh
        pip uninstall -y transformer-engine
        mv ./sglang /sglang
    " || { docker rm -f $RUN_ID; exit 1; }

docker commit $RUN_ID areal-env:latest
docker rm -f $RUN_ID
