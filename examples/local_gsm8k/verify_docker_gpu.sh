#!/bin/bash
# Verification script for Docker + GPU setup in WSL2

echo "=== Step 1: Check Docker is accessible ==="
docker --version
if [ $? -eq 0 ]; then
    echo "[OK] Docker is accessible"
else
    echo "[FAIL] Docker not found. Make sure Docker Desktop is running on Windows."
    exit 1
fi

echo ""
echo "=== Step 2: Check nvidia-smi in WSL2 ==="
nvidia-smi
if [ $? -eq 0 ]; then
    echo "[OK] GPU is visible in WSL2"
else
    echo "[FAIL] nvidia-smi failed. Check NVIDIA drivers on Windows."
    exit 1
fi

echo ""
echo "=== Step 3: Test Docker GPU access ==="
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Docker can access GPU!"
else
    echo "[FAIL] Docker cannot access GPU. Check NVIDIA Container Toolkit configuration."
    exit 1
fi

echo ""
echo "=== All checks passed! ==="
echo "You're ready to run GRPO training in Docker."

