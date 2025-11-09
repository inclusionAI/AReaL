#!/bin/bash
# Script to run AReaL GRPO Docker container on macOS (CPU-only mode)
# This will run without GPU support, suitable for testing on macOS

set -e

CONTAINER_NAME="areal-grpo-container"
SHARED_MEMORY="4g"  # Smaller for macOS
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "========================================"
echo "AReaL GRPO Docker Setup (macOS CPU)"
echo "========================================"
echo ""

# Check if Docker is running
echo "Checking Docker..."
if ! docker ps > /dev/null 2>&1; then
    echo "✗ Docker is not running. Please start Docker Desktop."
    exit 1
fi
echo "✓ Docker is running"

# Check if container already exists
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo ""
    echo "Container '$CONTAINER_NAME' already exists."
    echo "Options:"
    echo "  1. Start existing: docker start -i $CONTAINER_NAME"
    echo "  2. Remove and recreate: docker rm -f $CONTAINER_NAME && $0"
    exit 0
fi

# Build Docker image if it doesn't exist
echo ""
echo "Building Docker image..."
if ! docker images --format '{{.Repository}}:{{.Tag}}' | grep -q "^areal-grpo:local$"; then
    docker build -t areal-grpo:local -f "$(dirname "${BASH_SOURCE[0]}")/Dockerfile" "$(dirname "${BASH_SOURCE[0]}")"
    echo "✓ Docker image built"
else
    echo "✓ Docker image already exists"
fi

# Run Docker container (CPU-only mode)
echo ""
echo "Starting Docker container (CPU-only mode)..."
echo "Container name: $CONTAINER_NAME"
echo "Shared memory: $SHARED_MEMORY"
echo "Project directory: $PROJECT_DIR"
echo ""
echo "Note: This runs in CPU-only mode. GRPO training will be slower."
echo "      For GPU training, use Windows 11 PC with CUDA."
echo ""

docker run -it --name "$CONTAINER_NAME" \
    --shm-size="$SHARED_MEMORY" \
    --network host \
    -v "$PROJECT_DIR":/workspace/AReaL:rw \
    -v "$(dirname "${BASH_SOURCE[0]}")/wandb":/workspace/AReaL/examples/local_gsm8k/wandb:rw \
    -v "$(dirname "${BASH_SOURCE[0]}")/outputs":/workspace/AReaL/examples/local_gsm8k/outputs:rw \
    -w /workspace/AReaL \
    -e PYTHONPATH=/workspace/AReaL \
    -e CUDA_VISIBLE_DEVICES="" \
    -e WANDB_MODE=disabled \
    areal-grpo:local \
    /bin/bash

if [ $? -ne 0 ]; then
    echo ""
    echo "✗ Failed to start container"
    exit 1
fi

