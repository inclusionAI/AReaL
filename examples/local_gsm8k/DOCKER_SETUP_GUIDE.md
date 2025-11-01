# AReaL GRPO Docker Setup Guide

This guide explains how to run AReaL GRPO training in Docker on both **Windows 11 with CUDA GPU** and **macOS (CPU-only)**.

---

## Prerequisites

### For Windows 11 (CUDA GPU):
1. ✅ **Docker Desktop for Windows** with WSL2 backend
2. ✅ **NVIDIA Container Toolkit** (enables GPU access in Docker)
3. ✅ **CUDA-capable GPU** with drivers installed
4. ✅ **WSL2** installed and updated

### For macOS (CPU-only):
1. ✅ **Docker Desktop for Mac**
2. ⚠️ Note: No GPU support (will run on CPU)

---

## Quick Start

### Windows 11 (CUDA GPU)

1. **Open PowerShell** in the `examples/local_gsm8k/` directory

2. **Run the setup script**:
   ```powershell
   .\docker-run-windows.ps1
   ```

3. **If container already exists**, you can:
   ```powershell
   # Start existing container
   docker start -i areal-grpo-container
   
   # Or remove and recreate
   .\docker-run-windows.ps1 -RemoveExisting
   ```

4. **Inside the container**, verify setup:
   ```bash
   # Test environment
   python examples/local_gsm8k/test_grpo_docker.py
   
   # Check GPU
   nvidia-smi
   
   # Run GRPO training
   python -m areal.launcher.local \
       examples/local_gsm8k/train_grpo.py \
       --config examples/local_gsm8k/train_grpo.yaml \
       experiment_name=test \
       trial_name=t1
   ```

### macOS (CPU-only)

1. **Open Terminal** in the `examples/local_gsm8k/` directory

2. **Make script executable** (first time only):
   ```bash
   chmod +x docker-run-macos.sh
   ```

3. **Run the setup script**:
   ```bash
   ./docker-run-macos.sh
   ```

4. **Inside the container**, verify setup:
   ```bash
   # Test environment
   python examples/local_gsm8k/test_grpo_docker.py
   
   # Note: Training will be slow on CPU
   # Consider running only tests or small experiments
   ```

---

## Manual Docker Commands

If you prefer manual control, use these commands:

### Windows 11 (CUDA GPU)

```powershell
# Build image
docker build -t areal-grpo:local -f examples/local_gsm8k/Dockerfile examples/local_gsm8k

# Run container
docker run -it --name areal-grpo-container `
    --gpus all `
    --shm-size=8g `
    --network host `
    -v C:\Users\$env:USERNAME\GT\CS7643_Deep_Learning\ProjectLLM\AReaL:/workspace/AReaL:rw `
    -v ${PWD}\wandb:/workspace/AReaL/examples/local_gsm8k/wandb:rw `
    -v ${PWD}\outputs:/workspace/AReaL/examples/local_gsm8k/outputs:rw `
    -w /workspace/AReaL `
    -e PYTHONPATH=/workspace/AReaL `
    areal-grpo:local `
    /bin/bash
```

### macOS (CPU-only)

```bash
# Build image
docker build -t areal-grpo:local -f examples/local_gsm8k/Dockerfile examples/local_gsm8k

# Run container
docker run -it --name areal-grpo-container \
    --shm-size=4g \
    --network host \
    -v $(pwd)/../..:/workspace/AReaL:rw \
    -v $(pwd)/wandb:/workspace/AReaL/examples/local_gsm8k/wandb:rw \
    -v $(pwd)/outputs:/workspace/AReaL/examples/local_gsm8k/outputs:rw \
    -w /workspace/AReaL \
    -e PYTHONPATH=/workspace/AReaL \
    -e CUDA_VISIBLE_DEVICES="" \
    areal-grpo:local \
    /bin/bash
```

---

## Container Management

### Start/Stop Container

```bash
# Start existing container
docker start -i areal-grpo-container

# Stop container
docker stop areal-grpo-container

# Remove container
docker rm areal-grpo-container

# View logs
docker logs areal-grpo-container

# Execute command in running container
docker exec -it areal-grpo-container bash
```

### Access Container Shell

```bash
# Interactive shell (if container is running)
docker exec -it areal-grpo-container /bin/bash

# Interactive shell (starts container if stopped)
docker start -i areal-grpo-container
```

---

## Testing the Setup

### Run Test Script

Inside the container:

```bash
cd /workspace/AReaL
python examples/local_gsm8k/test_grpo_docker.py
```

This will verify:
- ✅ All Python dependencies are installed
- ✅ SGLang is available
- ✅ GRPO components can be imported
- ✅ Configuration can be loaded
- ✅ GPU/CPU detection

### Expected Output

**Windows (with GPU)**:
```
✓ CUDA available: NVIDIA GeForce RTX 4080 SUPER
✓ CUDA version: 12.x
✓ Number of GPUs: 1
```

**macOS (CPU-only)**:
```
⚠ CUDA not available - will use CPU
⚠ CPU training will be much slower than GPU
```

---

## Running GRPO Training

### Inside Container

```bash
cd /workspace/AReaL

# Load W&B API key (if using)
export WANDB_API_KEY=$(cat examples/local_gsm8k/wandb/api_key.txt)

# Run GRPO training
python -m areal.launcher.local \
    examples/local_gsm8k/train_grpo.py \
    --config examples/local_gsm8k/train_grpo.yaml \
    experiment_name=docker-test \
    trial_name=t1 \
    total_train_epochs=1 \
    train_dataset.batch_size=32
```

### Configuration Tips

For **testing**, modify `train_grpo.yaml` or use CLI overrides:

```bash
# Small test run (few samples, 1 epoch)
python -m areal.launcher.local \
    examples/local_gsm8k/train_grpo.py \
    --config examples/local_gsm8k/train_grpo.yaml \
    experiment_name=test \
    trial_name=small \
    total_train_epochs=1 \
    train_dataset.batch_size=16 \
    max_samples=100
```

---

## Troubleshooting

### Windows: GPU Not Available

**Problem**: `docker run` shows no GPU available

**Solutions**:
1. Ensure **NVIDIA Container Toolkit** is installed:
   ```powershell
   # Check if toolkit is installed
   docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
   ```

2. Verify **WSL2 backend** in Docker Desktop:
   - Settings → General → Use WSL 2 based engine

3. Update **NVIDIA drivers**:
   - Download from NVIDIA website

### macOS: Container Won't Start

**Problem**: Docker container fails to start

**Solutions**:
1. Increase Docker Desktop resources:
   - Settings → Resources → Memory (set to 8GB+)

2. Check Docker logs:
   ```bash
   docker logs areal-grpo-container
   ```

### SGLang Server Won't Start

**Problem**: SGLang inference server fails

**Possible causes**:
- GPU memory insufficient → Reduce batch size
- Port conflicts → Check `--network host` is used
- Missing dependencies → Run `pip install -e .` inside container

**Debug**:
```bash
# Inside container
cd /workspace/AReaL
python -c "import sglang; print(sglang.__version__)"
```

### Permission Errors

**Problem**: Permission denied when accessing mounted volumes

**Solution**: Ensure Docker Desktop has access to the mounted directories:
- Docker Desktop → Settings → Resources → File Sharing

---

## File Structure

```
examples/local_gsm8k/
├── Dockerfile                    # Docker image definition
├── docker-compose.yml            # Docker Compose config (optional)
├── docker-run-windows.ps1       # Windows setup script
├── docker-run-macos.sh          # macOS setup script
├── test_grpo_docker.py          # Environment test script
├── requirements-docker.txt      # Additional Python deps
└── DOCKER_SETUP_GUIDE.md        # This file
```

---

## Mounted Volumes

The Docker container mounts:

1. **Project code**: `/workspace/AReaL` ← Your local AReaL repo
2. **W&B keys**: `./wandb` ← API keys (gitignored)
3. **Outputs**: `./outputs` ← Training outputs, logs, checkpoints
4. **Models** (optional): Mount your model cache for faster startup

---

## Performance Notes

### Windows 11 (CUDA GPU)
- ✅ **Full GPU acceleration** available
- ✅ **Production training** possible
- ✅ **Fast inference** with SGLang
- ⚠️ **WSL2 overhead**: ~5-10% slower than native Linux

### macOS (CPU-only)
- ⚠️ **CPU-only**: Much slower than GPU
- ⚠️ **SGLang may not work** (designed for CUDA)
- ✅ **Good for testing** setup and code
- ✅ **Use for small experiments** only

---

## Next Steps

1. ✅ **Test the setup**: Run `test_grpo_docker.py`
2. ✅ **Run small training**: 1 epoch, 10 samples
3. ✅ **Check outputs**: Verify logs and checkpoints are saved
4. ✅ **Scale up**: Increase batch size, epochs on GPU system

---

## References

- **AReaL Official Docs**: `docs/tutorial/installation.md`
- **AReaL Docker Image**: `ghcr.io/inclusionai/areal-runtime:v0.3.4`
- **NVIDIA Container Toolkit**: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/

