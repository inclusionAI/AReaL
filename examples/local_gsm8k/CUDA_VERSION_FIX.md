# CUDA Version Mismatch Fix

## The Problem

Your system has:
- **CUDA 12.6** (from `nvidia-smi`)
- **Driver Version 560.94**

The official AReaL Docker image requires:
- **CUDA 12.9+**

Error message:
```
nvidia-container-cli: requirement error: unsatisfied condition: cuda>=12.9
```

## Solution Options

### ✅ Option 1: Update NVIDIA Drivers (Easiest - 15 minutes)

This is the **recommended** solution:

1. **Download latest drivers**:
   - Go to: https://www.nvidia.com/Download/index.aspx
   - Product: **GeForce RTX 4080 SUPER**
   - Operating System: **Windows 64-bit**
   - Download Type: **Game Ready Driver** (or Studio Driver)
   - Click "Search" and download

2. **Install**:
   - Run the downloaded installer
   - Choose "**Custom installation**" → "**Clean install**" (removes old drivers)
   - Let it complete and restart

3. **Verify**:
   ```powershell
   # In PowerShell (Windows)
   nvidia-smi
   ```
   Should show `CUDA Version: 12.9` or higher

4. **Restart Docker Desktop**:
   - Close Docker Desktop completely
   - Restart it
   - Test again

5. **Retry Docker run**:
   ```bash
   # In WSL2
   docker run -it --name areal-grpo \
       --gpus all \
       --ipc=host \
       --shm-size=16g \
       -v /mnt/c/Users/tongz/git/GT/AReaL:/workspace/AReaL \
       -w /workspace/AReaL \
       ghcr.io/inclusionai/areal-runtime:v0.3.4 \
       /bin/bash
   ```

### Option 2: Use CUDA 12.6 Compatible Base Image (Alternative - 30+ minutes)

If you **cannot** update drivers right now, you can build a custom Docker image:

#### Step 1: Check what base image SGLang uses

```bash
# Check available SGLang tags with CUDA 12.6
docker search sglang
# Or visit: https://hub.docker.com/r/lmsysorg/sglang/tags
```

#### Step 2: Modify Dockerfile

The current `examples/local_gsm8k/Dockerfile` uses:
```dockerfile
FROM ghcr.io/inclusionai/areal-runtime:v0.3.4
```

We need to check if there's a CUDA 12.6-compatible version or build from a CUDA 12.6 base.

**Alternative approach**: Use `nvidia/cuda:12.6.0-runtime-ubuntu22.04` as base and install dependencies manually:

```dockerfile
FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04

# Install Python and basic tools
RUN apt-get update && apt-get install -y \
    python3.12 python3.12-dev python3-pip git \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Install PyTorch with CUDA 12.6
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Install other dependencies...
# (This becomes quite complex - see Option 1 instead)
```

**Note**: This requires rebuilding all dependencies, which is time-consuming.

### Option 3: Use CPU-Only Mode (For Testing Only - Not for Training)

If you just want to test the setup without GPU:

```bash
# Remove --gpus all flag
docker run -it --name areal-grpo \
    --ipc=host \
    --shm-size=16g \
    -v /mnt/c/Users/tongz/git/GT/AReaL:/workspace/AReaL \
    -w /workspace/AReaL \
    ghcr.io/inclusionai/areal-runtime:v0.3.4 \
    /bin/bash
```

**Warning**: Training will be **extremely slow** on CPU. Not recommended for actual training.

---

## Recommended: Update Drivers

**Option 1 (Update Drivers) is strongly recommended** because:
- ✅ Quickest solution (~15 minutes)
- ✅ Gets you latest GPU performance improvements
- ✅ Works with official pre-built images
- ✅ No custom Docker builds needed
- ✅ Future-proof for newer CUDA requirements

---

## After Driver Update

Once drivers are updated:

1. **Verify CUDA version**:
   ```powershell
   nvidia-smi
   ```

2. **Restart WSL2**:
   ```powershell
   wsl --shutdown
   # Then reopen Ubuntu
   ```

3. **Test GPU access**:
   ```bash
   # In WSL2
   nvidia-smi
   docker run --rm --gpus all nvidia/cuda:12.9.0-base-ubuntu22.04 nvidia-smi
   ```

4. **Run AReaL container**:
   ```bash
   docker run -it --name areal-grpo \
       --gpus all \
       --ipc=host \
       --shm-size=16g \
       -v /mnt/c/Users/tongz/git/GT/AReaL:/workspace/AReaL \
       -w /workspace/AReaL \
       ghcr.io/inclusionai/areal-runtime:v0.3.4 \
       /bin/bash
   ```

---

## Troubleshooting

### "Still getting CUDA 12.9 requirement error after driver update"

1. **Verify driver actually updated**:
   ```powershell
   nvidia-smi
   ```
   Check the "CUDA Version" line

2. **Restart everything**:
   ```powershell
   # Shutdown WSL2
   wsl --shutdown
   # Restart Docker Desktop
   # Then reopen WSL2
   ```

3. **Clear Docker GPU cache** (if needed):
   ```bash
   docker system prune -a
   ```

### "Can't update drivers (work computer, restrictions, etc.)"

If you absolutely cannot update drivers:
- Build custom Docker image (Option 2 above)
- Or use a different machine/VPS with CUDA 12.9+ support
- Or wait until you can update drivers

---

## Quick Command Reference

After driver update:
```bash
# Test
docker run --rm --gpus all nvidia/cuda:12.9.0-base-ubuntu22.04 nvidia-smi

# Run AReaL
docker run -it --name areal-grpo \
    --gpus all --ipc=host --shm-size=16g \
    -v /mnt/c/Users/tongz/git/GT/AReaL:/workspace/AReaL \
    -w /workspace/AReaL \
    ghcr.io/inclusionai/areal-runtime:v0.3.4 /bin/bash
```

