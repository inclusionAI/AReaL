#!/usr/bin/env bash

set -e

GIT_REPO_URL=${GIT_REPO_URL:?"GIT_REPO_URL is not set"}
GIT_COMMIT_SHA=${GIT_COMMIT_SHA:?"GIT_COMMIT_SHA is not set"}
RUN_ID="areal-$(date +%s%N)"

echo "GIT_REPO_URL: $GIT_REPO_URL"
echo "GIT_COMMIT_SHA: $GIT_COMMIT_SHA"

mkdir -p /tmp/pip-cache

mkdir -p "/tmp/$RUN_ID"
cd "/tmp/$RUN_ID"

git clone "$GIT_REPO_URL" --depth 1 --branch "$GIT_COMMIT_SHA" --single-branch .

if docker ps -a --format '{{.Names}}' | grep -q "$RUN_ID"; then
    docker rm -f $RUN_ID
fi

docker run -d \
    --name $RUN_ID \
    --gpus all \
    --shm-size=8g \
    -v $(pwd):/workspace \
    -v /tmp/pip-cache:/root/.cache/pip \
    -w /workspace \
    nvcr.io/nvidia/pytorch:25.01-py3 \
    sleep 3600

docker exec $RUN_ID bash -c "
    python -m pip install --upgrade pip
    pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
    pip config unset global.extra-index-url
    bash examples/env/scripts/setup-pip-deps.sh
    python -m pytest arealite/
"

docker rm -f $RUN_ID
