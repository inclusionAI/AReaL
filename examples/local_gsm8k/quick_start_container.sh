#!/bin/bash
# Quick start script for AReaL GRPO Docker container

# Remove existing container if it exists
docker rm -f areal-grpo 2>/dev/null

# Run new container
docker run -it --name areal-grpo \
    --gpus all \
    --ipc=host \
    --shm-size=16g \
    -v /mnt/c/Users/tongz/git/GT/AReaL:/workspace/AReaL \
    -w /workspace/AReaL \
    ghcr.io/inclusionai/areal-runtime:v0.3.4 \
    /bin/bash

