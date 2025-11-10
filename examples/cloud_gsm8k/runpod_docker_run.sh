#!/bin/bash
# RunPod-optimized Docker run script
# Use this inside a RunPod pod to start the training container
#
# Usage:
#   bash examples/cloud_gsm8k/runpod_docker_run.sh

set -e

# Configuration
CONTAINER_NAME="${CONTAINER_NAME:-areal-grpo-runpod}"
PROJECT_PATH="${PROJECT_PATH:-/workspace/AReaL}"
OUTPUTS_PATH="${OUTPUTS_PATH:-/workspace/outputs}"
SHARED_MEMORY="${SHARED_MEMORY:-16g}"
IMAGE_NAME="ghcr.io/inclusionai/areal-runtime:v0.3.4"

# Check WandB API key
if [ -z "$WANDB_API_KEY" ]; then
    echo "ERROR: WANDB_API_KEY environment variable not set"
    echo "Set it in RunPod pod environment variables or:"
    echo "  export WANDB_API_KEY=your-api-key"
    exit 1
fi

# Check if container already exists
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Container '$CONTAINER_NAME' already exists."
    echo "Options:"
    echo "  1. Remove existing: docker rm -f $CONTAINER_NAME"
    echo "  2. Start existing: docker start -ai $CONTAINER_NAME"
    exit 1
fi

# Create outputs directory if it doesn't exist
mkdir -p "$OUTPUTS_PATH"

echo "=========================================="
echo "RunPod AReaL Docker Container"
echo "=========================================="
echo "Container name: $CONTAINER_NAME"
echo "Project path: $PROJECT_PATH"
echo "Outputs path: $OUTPUTS_PATH (RunPod network volume)"
echo "Shared memory: $SHARED_MEMORY"
echo "Image: $IMAGE_NAME"
echo "=========================================="
echo ""

# Pull image if not exists
echo "Checking Docker image..."
if ! docker images --format '{{.Repository}}:{{.Tag}}' | grep -q "^${IMAGE_NAME}$"; then
    echo "Pulling Docker image: $IMAGE_NAME"
    docker pull "$IMAGE_NAME"
else
    echo "Docker image found locally"
fi

echo ""
echo "Starting container..."
echo ""

# RunPod-optimized Docker run command
# Note: RunPod pods already have GPU access, so --gpus all may not be needed
# But we include it for compatibility
docker run -it --name "$CONTAINER_NAME" \
    --gpus all \
    --ipc=host \
    --shm-size="$SHARED_MEMORY" \
    --network host \
    -e WANDB_API_KEY="$WANDB_API_KEY" \
    -e PYTHONPATH="$PROJECT_PATH" \
    -e CUDA_VISIBLE_DEVICES=0 \
    -v "$PROJECT_PATH:$PROJECT_PATH:rw" \
    -v "$OUTPUTS_PATH:$OUTPUTS_PATH:rw" \
    -w "$PROJECT_PATH" \
    "$IMAGE_NAME" \
    /bin/bash

echo ""
echo "Container started. You should now be inside the container."
echo ""
echo "Next steps:"
echo "  1. cd $PROJECT_PATH"
echo "  2. git clone -b DL4Math https://github.com/nexthybrid/AReaL.git .  (if not already cloned)"
echo "  3. pip install -e ."
echo "  4. bash examples/cloud_gsm8k/run_training_cloud.sh 1hour"

