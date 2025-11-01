# Running AReaL GRPO Training on Windows 11 with Docker + GPU

This guide explains how to run AReaL GRPO training on Windows 11 using Docker while leveraging your NVIDIA RTX 4080 SUPER GPU.

## Why Docker?

AReaL's GRPO training requires:
- Linux environment (for SGLang and various dependencies)
- SGLang inference server (difficult to install natively on Windows)
- Unix-style commands in the launcher

Docker solves all these issues by providing a Linux environment that can access your Windows GPU.

## Prerequisites

### 1. Install WSL2

Windows Subsystem for Linux 2 is required for Docker Desktop:

```powershell
# Run PowerShell as Administrator
wsl --install
# Restart your computer after installation
```

Verify installation:
```powershell
wsl --list --verbose
```

### 2. Install Docker Desktop for Windows

1. Download from: https://www.docker.com/products/docker-desktop/
2. Install Docker Desktop
3. During installation, ensure "Use WSL 2 instead of Hyper-V" is selected
4. After installation, go to **Settings → General** and enable:
   - ✅ Use the WSL 2 based engine
   - ✅ Start Docker Desktop when you log in

### 3. Install NVIDIA Container Toolkit on WSL2

The NVIDIA Container Toolkit allows Docker containers to access your GPU.

#### Step 1: Open WSL2 Ubuntu Terminal

```powershell
wsl
# You should now be in Ubuntu Linux
```

#### Step 2: Install NVIDIA Container Toolkit in WSL2

```bash
# Add NVIDIA Container Toolkit repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install the toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker to use the runtime
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker daemon (in WSL2)
sudo service docker restart
```

#### Step 3: Verify GPU Access in WSL2

```bash
# Check NVIDIA driver in WSL2
nvidia-smi
```

You should see your RTX 4080 SUPER listed. If not, ensure you have:
- Latest NVIDIA drivers installed on Windows
- WSL2 with CUDA support (Windows 11 should have this by default)

### 4. Install NVIDIA Container Toolkit on Windows Host

The Windows Docker Desktop also needs GPU support. Download and install:
- NVIDIA Container Toolkit for Windows (if available) or
- Ensure Docker Desktop can access NVIDIA GPU

**Note**: Docker Desktop on Windows with GPU support is still evolving. You may need to:
1. Use Windows Insider builds with WSLg GPU support, OR
2. Use Docker Desktop with WSL2 backend (recommended)

## Docker Setup

### Option 1: Use Pre-built AReaL Image (Recommended)

```bash
# In WSL2 Ubuntu terminal
cd /mnt/c/Users/tongz/git/GT/AReaL

# Pull the pre-built image
docker pull ghcr.io/inclusionai/areal-runtime:v0.3.4

# Run container with GPU access
docker run -it --name areal-training \
    --gpus all \
    --ipc=host \
    --shm-size=16g \
    -v $(pwd):/workspace/AReaL \
    -v /home/$USER/.cache:/root/.cache \
    -w /workspace/AReaL \
    ghcr.io/inclusionai/areal-runtime:v0.3.4 \
    /bin/bash
```

### Option 2: Build from Dockerfile

If you need to customize the image:

```bash
# In WSL2 Ubuntu terminal
cd /mnt/c/Users/tongz/git/GT/AReaL

# Build the Docker image
docker build -t areal-local:latest -f Dockerfile .

# Run container
docker run -it --name areal-training \
    --gpus all \
    --ipc=host \
    --shm-size=16g \
    -v $(pwd):/workspace/AReaL \
    -v /home/$USER/.cache:/root/.cache \
    -w /workspace/AReaL \
    areal-local:latest \
    /bin/bash
```

**Note**: Building from Dockerfile takes 30-60 minutes and requires significant disk space (~50GB).

## Running GRPO Training in Docker

Once inside the container:

```bash
# Install AReaL (if using pre-built image)
cd /workspace/AReaL
pip install -e .

# Navigate to your training directory
cd examples/local_gsm8k

# Set WandB API key (optional)
export WANDB_API_KEY=5cd583e967c0e092a7f7be82e0479c1f71eeeab9

# Run GRPO training
python -m areal.launcher.local train_grpo.py \
    --config train_grpo.yaml \
    experiment_name=gsm8k-grpo-local \
    trial_name=trial0
```

## Alternative: Using WSL2 Directly (Without Docker)

You can also install everything directly in WSL2:

```bash
# In WSL2 Ubuntu terminal
cd /mnt/c/Users/tongz/git/GT/AReaL

# Install Python dependencies
sudo apt update
sudo apt install python3 python3-pip python3-venv

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install SGLang (this should work in WSL2)
pip install sglang[all]

# Install other dependencies
pip install -r requirements.txt

# Install AReaL
pip install -e .

# Run training
cd examples/local_gsm8k
python -m areal.launcher.local train_grpo.py --config train_grpo.yaml
```

## Troubleshooting

### GPU Not Detected in Container

```bash
# Test GPU access
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

If this fails:
1. Verify NVIDIA drivers are installed on Windows
2. Check `nvidia-smi` works in WSL2: `wsl nvidia-smi`
3. Restart Docker Desktop
4. Check Docker Desktop → Settings → Resources → WSL Integration is enabled

### Out of Memory Errors

Increase shared memory:
```bash
docker run --shm-size=32g ...  # Increase from 16g to 32g or higher
```

### Permission Issues

```bash
# Fix permissions for mounted volumes
sudo chown -R $USER:$USER /mnt/c/Users/tongz/git/GT/AReaL
```

### Port Conflicts

If ports are already in use, modify the allocation mode in `train_grpo.yaml`:
```yaml
allocation_mode: sglang.d1p1t1+d1p1t1  # Single GPU mode
```

## Performance Tips

1. **Use WSL2 backend**: Better performance than Hyper-V
2. **Store data in WSL2 filesystem**: Accessing `/mnt/c/` is slower
   ```bash
   # Copy data to WSL2 filesystem
   cp -r /mnt/c/Users/tongz/git/GT/AReaL ~/AReaL
   cd ~/AReaL
   ```
3. **Increase Docker resources**: Docker Desktop → Settings → Resources
   - CPUs: Use all available
   - Memory: At least 16GB (more if available)
   - Swap: 2GB

## Quick Reference Commands

```bash
# Start existing container
docker start areal-training
docker exec -it areal-training /bin/bash

# Stop container
docker stop areal-training

# Remove container
docker rm areal-training

# View container logs
docker logs areal-training

# Check GPU in container
docker exec areal-training nvidia-smi

# Clean up Docker (remove unused images/containers)
docker system prune -a
```

## Expected Performance

With RTX 4080 SUPER (16GB VRAM):
- Can train Qwen 0.5B-1.5B models
- Batch size: 4-8 depending on model size
- Training speed: ~2-5 steps/second for GRPO
- Memory: ~12-14GB VRAM usage during training

## Next Steps

1. Test GPU access: `nvidia-smi` inside container
2. Run a quick test: Start with 1 epoch, small batch size
3. Monitor training: Use WandB dashboard
4. Scale up: Once working, increase epochs/batch size

## Resources

- [NVIDIA Container Toolkit Docs](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- [Docker Desktop WSL2 Backend](https://docs.docker.com/desktop/windows/wsl/)
- [WSL2 GPU Support](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
