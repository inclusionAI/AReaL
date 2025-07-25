#!/usr/bin/env bash

set -e

ls

RUN_ID="test-arealite-$RANDOM"

# Calculate environment hash from pyproject.toml
ENV_SHA=$(sha256sum pyproject.toml | awk '{print $1}')

# Build environment image if it doesn't exist
if ! docker images --format '{{.Repository}}:{{.Tag}}' | grep -q "areal-env:$ENV_SHA"; then
    echo "Building image areal-env:$ENV_SHA..."

    # Build the environment image
    docker run \
        --name $RUN_ID \
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

    # Commit the container as the environment image
    docker commit $RUN_ID "areal-env:$ENV_SHA"
    docker rm -f $RUN_ID
    
    echo "Image areal-env:$ENV_SHA built successfully."
else
    echo "Image areal-env:$ENV_SHA already exists, skipping build."
fi

# Run tests using the environment image
echo "Running tests on image areal-env:$ENV_SHA..."
docker run \
    --name $RUN_ID \
    --gpus all \
    --shm-size=8g \
    -v $(pwd):/workspace \
    -w /workspace \
    "areal-env:$ENV_SHA" \
    bash -c "
        HF_ENDPOINT=https://hf-mirror.com python -m pytest -s arealite/
    " || { docker rm -f $RUN_ID; exit 1; }

docker rm -f $RUN_ID
