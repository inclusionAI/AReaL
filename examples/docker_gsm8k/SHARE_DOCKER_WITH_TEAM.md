# How to Share Docker Setup with Team Members

## Quick Start Guide

This guide explains how to share your Docker-based GRPO training environment with team members so they can run the same training setup on their machines.

## What Gets Shared?

### ✅ Code Files (in Git Repository)
These files are automatically shared when team members clone the repo:
- `examples/docker_gsm8k/gsm8k_grpo.py` - Training script
- `examples/docker_gsm8k/gsm8k_grpo.yaml` - Configuration
- `examples/docker_gsm8k/README.md` - Basic docs
- `examples/docker_gsm8k/run_training.sh` - Training launcher
- `examples/docker_gsm8k/TEAM_COLLABORATION.md` - Collaboration guide

### ❌ Not Shared (Each Person Sets Up Individually)
- Docker Desktop installation
- NVIDIA drivers
- WandB API key (personal)
- Personal paths and configurations

## Step-by-Step Sharing Process

### For You (Team Lead/First User)

1. **Ensure all Docker setup files are committed:**
   ```bash
   git status
   git add examples/docker_gsm8k/
   git commit -m "feat: add Docker-based GRPO training setup"
   git push
   ```

2. **Share the repository** (if not already shared):
   - GitHub/GitLab/Bitbucket link
   - Or internal git server

3. **Create a quick start document** (this file!)

### For Team Members

Each team member follows these steps:

#### Step 1: Clone Repository

```bash
git clone <your-repo-url>
cd AReaL
```

#### Step 2: Install Docker Desktop

1. Download: https://www.docker.com/products/docker-desktop
2. Install Docker Desktop
3. Enable WSL2 integration:
   - Open Docker Desktop
   - Go to Settings → Resources → WSL Integration
   - Enable integration for your WSL distro (Ubuntu)
   - Click "Apply & Restart"

#### Step 3: Install NVIDIA Container Toolkit (for GPU support)

For Windows 11 with WSL2:

1. **Install NVIDIA Driver** (if not installed):
   - Download from: https://www.nvidia.com/Download/index.aspx
   - Install CUDA 13.0+ compatible driver

2. **Install NVIDIA Container Toolkit in WSL2:**
   ```bash
   # In WSL2 (Ubuntu)
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker  # Restart Docker Desktop after this
   ```

3. **Verify GPU access:**
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
   ```

#### Step 4: Set Up WandB API Key

```bash
# Create wandb directory (if it doesn't exist)
mkdir -p examples/local_gsm8k/wandb

# Add your personal WandB API key
echo "your-wandb-api-key-here" > examples/local_gsm8k/wandb/.wandb_api_key

# Verify (this file should NOT be committed to git)
cat examples/local_gsm8k/wandb/.wandb_api_key
```

#### Step 5: Start Docker Container

**Important**: Each person needs to adjust the path to their local AReaL directory.

```bash
# In WSL2 or PowerShell
cd /mnt/c/Users/YOUR-USERNAME/path/to/AReaL

# Get your Windows path, convert to WSL path:
# Windows: C:\Users\alice\Documents\AReaL
# WSL:     /mnt/c/Users/alice/Documents/AReaL

# Start container (adjust YOUR-USERNAME and path)
docker run -it --name areal-grpo \
    --gpus all \
    --ipc=host \
    --shm-size=16g \
    -v /mnt/c/Users/YOUR-USERNAME/path/to/AReaL:/workspace/AReaL:rw \
    -w /workspace/AReaL \
    ghcr.io/inclusionai/areal-runtime:v0.3.4 \
    /bin/bash
```

**Quick Path Finder:**
```bash
# In WSL2, find your project path:
pwd
# Then use that path in the -v flag above
```

#### Step 6: Test the Setup

```bash
# Inside Docker container
cd /workspace/AReaL

# Check GPU access
nvidia-smi

# Verify Python and dependencies
python3 --version

# Test import
python3 -c "import torch; print(torch.cuda.is_available())"
```

#### Step 7: Run Training

```bash
# Inside Docker container
cd /workspace/AReaL

# Set WandB API key
export WANDB_API_KEY=$(cat examples/local_gsm8k/wandb/.wandb_api_key 2>/dev/null || echo "your-api-key")

# Run training (use unique trial_name for each team member)
python3 -m areal.launcher.local examples/docker_gsm8k/gsm8k_grpo.py \
    --config examples/docker_gsm8k/gsm8k_grpo.yaml \
    experiment_name=gsm8k-grpo-docker \
    trial_name=alice_trial0  # Use unique name!
```

## Sharing Methods

### Method 1: Git Repository (Recommended)

**Best for**: Code and configuration files

1. Commit Docker setup files to git:
   ```bash
   git add examples/docker_gsm8k/
   git commit -m "feat: add Docker GRPO training setup"
   git push
   ```

2. Team members clone and follow setup steps above.

**Pros:**
- ✅ Version controlled
- ✅ Easy updates
- ✅ Standard workflow

**Cons:**
- ❌ Each person still needs to install Docker
- ❌ Each person sets up WandB key individually

### Method 2: Docker Image Sharing (Optional)

If you've customized the Docker image, you can share it:

```bash
# Export your container as image
docker commit areal-grpo myteam/areal-grpo:latest

# Save to file
docker save myteam/areal-grpo:latest | gzip > areal-grpo.tar.gz

# Team members load it
docker load < areal-grpo.tar.gz
```

**Note**: Usually not needed since we use the official AReaL image.

### Method 3: Docker Compose (Optional - For Advanced Users)

Create `docker-compose.yml` for easier sharing:

```yaml
version: '3.8'

services:
  areal-grpo:
    image: ghcr.io/inclusionai/areal-runtime:v0.3.4
    container_name: areal-grpo
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    volumes:
      - ${PWD}:/workspace/AReaL:rw
    working_dir: /workspace/AReaL
    shm_size: 16g
    ipc: host
    stdin_open: true
    tty: true
```

Then team members just run:
```bash
docker-compose up -d
docker-compose exec areal-grpo bash
```

## Team Collaboration Best Practices

### 1. Use Unique Trial Names

Each team member should use a unique `trial_name` to avoid conflicts:

```bash
# Alice
trial_name=alice_trial0

# Bob  
trial_name=bob_trial0

# Charlie
trial_name=charlie_trial0
```

### 2. Shared WandB Project

All use the same WandB project for easy comparison:
- Project: `gsm8k-grpo-local` (already configured)
- Each person's runs appear separately

### 3. Code Updates

```bash
# Inside Docker container
cd /workspace/AReaL
git pull origin main  # Get latest code
# Changes are reflected immediately (volume mount)
```

### 4. Sharing Trained Models

**Option A: Git LFS** (for small models)
```bash
git lfs install
git lfs track "*.safetensors"
git add outputs/grpo/checkpoints/...
git commit -m "checkpoint: trained model"
git push
```

**Option B: WandB Artifacts** (recommended)
- Models automatically logged to WandB
- Team can download via WandB UI

**Option C: Shared Storage**
- Network drive
- Google Drive/Dropbox
- S3 bucket

## Troubleshooting Common Issues

### Issue: "Docker command not found in WSL"

**Solution**: Enable WSL integration in Docker Desktop:
- Docker Desktop → Settings → Resources → WSL Integration
- Enable your WSL distro
- Restart Docker Desktop

### Issue: "nvidia-container-cli: requirement error: unsatisfied condition: cuda>=12.9"

**Solution**: Update NVIDIA drivers:
- Download latest from NVIDIA website
- Install CUDA 13.0+ compatible driver

### Issue: "Container name already in use"

**Solution**: Remove existing container:
```bash
docker stop areal-grpo
docker rm areal-grpo
# Then run docker run again
```

### Issue: "Path not found" in volume mount

**Solution**: Use absolute WSL paths:
```bash
# In WSL2, check your path
cd ~/path/to/AReaL
pwd  # Use this exact path in -v flag
```

### Issue: "Permission denied" in container

**Solution**: Ensure volume mount has proper permissions:
```bash
# Use :rw (read-write) flag in docker run
-v /path:/workspace/AReaL:rw
```

## Checklist for Team Onboarding

- [ ] Docker Desktop installed
- [ ] WSL2 integration enabled
- [ ] NVIDIA Container Toolkit installed
- [ ] GPU access verified (`nvidia-smi` in container)
- [ ] Repository cloned
- [ ] WandB API key configured
- [ ] Docker container starts successfully
- [ ] Can run training script
- [ ] WandB logging works

## Quick Reference Commands

### Starting Container (Template)
```bash
docker run -it --name areal-grpo \
    --gpus all \
    --ipc=host \
    --shm-size=16g \
    -v /mnt/c/Users/USERNAME/path/to/AReaL:/workspace/AReaL:rw \
    -w /workspace/AReaL \
    ghcr.io/inclusionai/areal-runtime:v0.3.4 \
    /bin/bash
```

### Running Training
```bash
# Inside container
export WANDB_API_KEY=$(cat examples/local_gsm8k/wandb/.wandb_api_key)
python3 -m areal.launcher.local examples/docker_gsm8k/gsm8k_grpo.py \
    --config examples/docker_gsm8k/gsm8k_grpo.yaml \
    experiment_name=gsm8k-grpo-docker \
    trial_name=YOUR_NAME_trial0
```

### Updating Code
```bash
# Inside container or on host (same files!)
cd /workspace/AReaL
git pull
```

## Summary

**What to Share:**
1. Git repository with `examples/docker_gsm8k/` folder
2. This documentation

**What Each Person Does:**
1. Clone repo
2. Install Docker Desktop + WSL2 integration
3. Install NVIDIA Container Toolkit
4. Set WandB API key
5. Start container (with their path)
6. Run training

**Key Points:**
- ✅ Code is in git (easy to share)
- ✅ Docker image pulls automatically
- ✅ Each person needs unique `trial_name`
- ✅ Path in docker run command is personal
- ✅ WandB API key is personal (not in git)

---

For more details, see: `examples/docker_gsm8k/TEAM_COLLABORATION.md`

