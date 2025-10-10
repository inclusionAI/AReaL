#!/usr/bin/env bash
set -euo pipefail

WORKDIR=/storage/openpsi/users/lichangye.lcy/AReaL
IMAGE=/storage/openpsi/images/areal-v0.3.3-sglang-v0.5.2-vllm-v0.10.2-v2.sif

# 只启动一个交互容器；后续在容器内再算 IP、起 Ray
srun --mpi=pmi2 \
  --ntasks=1 \
  --gres=gpu:8 \
  --chdir="$WORKDIR" \
  --cpus-per-task=64 \
  --mem=1500G \
  --nodes=1 \
  --pty singularity shell \
    --nv --no-home --writable-tmpfs \
    --bind /storage:/storage \
    "$IMAGE"


