# Team Collaboration Guide for Docker Setup

## Overview

This guide explains how to share the Docker-based GRPO training setup with your team and how to use Git from inside the Docker container.

## Sharing Docker Setup with Team

### 1. What to Share

Your team needs:
- ✅ Docker Desktop installed (with WSL2 integration)
- ✅ NVIDIA Container Toolkit (for GPU support)
- ✅ The AReaL Docker image (automatically pulled)
- ✅ These configuration files in `examples/docker_gsm8k/`

### 2. Files Your Team Needs

Share these files with your team (already in git):
- `examples/docker_gsm8k/gsm8k_grpo.py` - Training script
- `examples/docker_gsm8k/gsm8k_grpo.yaml` - Configuration
- `examples/docker_gsm8k/README.md` - Instructions
- `examples/docker_gsm8k/run_training.sh` - Training launcher
- `examples/docker_gsm8k/HOW_TO_TEST_MODEL.md` - Testing guide
- `examples/docker_gsm8k/TEAM_COLLABORATION.md` - This file

### 3. Team Setup Steps

Each team member should:

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd AReaL
   ```

2. **Install Docker Desktop** (if not already installed):
   - Download from https://www.docker.com/products/docker-desktop
   - Enable WSL2 integration in Docker Desktop settings

3. **Install NVIDIA Container Toolkit** (for GPU support):
   - Follow instructions in `examples/local_gsm8k/WINDOWS_DOCKER_SETUP.md`
   - Or: `examples/local_gsm8k/DOCKER_WINDOWS_SETUP.md` if it exists

4. **Update their WandB API key**:
   ```bash
   # Create the file (not tracked in git)
   mkdir -p examples/local_gsm8k/wandb
   echo "their-wandb-api-key" > examples/local_gsm8k/wandb/.wandb_api_key
   ```

5. **Start the Docker container:**
   ```bash
   # In WSL2 or PowerShell
   wsl
   
   # Navigate to project
   cd /mnt/c/Users/their-username/path/to/AReaL
   
   # Start container (adjust path as needed)
   docker run -it --name areal-grpo \
       --gpus all \
       --ipc=host \
       --shm-size=16g \
       -v /mnt/c/Users/their-username/path/to/AReaL:/workspace/AReaL:rw \
       -w /workspace/AReaL \
       ghcr.io/inclusionai/areal-runtime:v0.3.4 \
       /bin/bash
   ```

6. **Run training:**
   ```bash
   # Inside container
   cd /workspace/AReaL
   export WANDB_API_KEY=$(cat examples/local_gsm8k/wandb/.wandb_api_key 2>/dev/null || echo "their-api-key")
   
   python3 -m areal.launcher.local examples/docker_gsm8k/gsm8k_grpo.py \
       --config examples/docker_gsm8k/gsm8k_grpo.yaml \
       experiment_name=gsm8k-grpo-docker \
       trial_name=trial0
   ```

## Using Git from Docker Container

### Yes, You Can Git Commit from Inside Docker! ✅

Since your workspace is **mounted as a volume** (`-v /path/to/AReaL:/workspace/AReaL`), the `.git` directory is accessible from inside the container. This means:

- ✅ Code changes in the container = changes on your host
- ✅ Git commands work inside the container
- ✅ Commits are made to your actual repository
- ✅ All git history and branches are accessible

### Important: Git Configuration

Before committing, **set your git identity** inside the container:

```bash
# Inside Docker container
git config user.name "Your Name"
git config user.email "your.email@example.com"

# Or use --global (but this only affects the container):
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Check your config
git config --list
```

### Example Workflow

```bash
# 1. Inside Docker container, make code changes
cd /workspace/AReaL
# ... edit files ...

# 2. Check status
git status

# 3. Stage changes
git add examples/docker_gsm8k/new_file.py

# 4. Commit
git commit -m "feat: add new training script for docker"

# 5. Push (from container or host, both work)
git push origin main
```

### Important Notes

1. **Same Repository**: Since the workspace is mounted, git operations affect the **same repository** on your host. You can also commit from your host machine - they're the same!

2. **Git Credentials**: If you need to authenticate (HTTPS push/pull), you may need to:
   ```bash
   # Inside container
   git config credential.helper store
   # Then push/pull and enter credentials once
   ```

3. **SSH Keys**: If using SSH authentication, you may need to:
   ```bash
   # Option 1: Mount SSH directory
   docker run ... \
       -v ~/.ssh:/root/.ssh:ro \
       ...
   
   # Option 2: Use HTTPS with credential helper
   ```

4. **Work from Host or Container**: Since files are shared, you can:
   - Edit code in Cursor/VS Code on Windows (host)
   - Commit from Docker container
   - Or vice versa - it's all the same filesystem!

### Recommended Approach

For best collaboration:

1. **Edit code on host** (using Cursor/VS Code) - better IDE support
2. **Test/run in Docker** - consistent environment
3. **Commit from either** - your choice!

Example:
```bash
# Edit in Cursor on Windows (host)
# Then test in Docker:
docker exec -it areal-grpo bash -c "cd /workspace/AReaL && python3 -m areal.launcher.local ..."

# Commit from host (easier):
git add .
git commit -m "feat: ..."
git push
```

## Sharing Trained Models

### Option 1: Checkpoints in Git (Large Files)

If you want to share trained models via git:

1. **Use Git LFS** (Large File Storage):
   ```bash
   # Install Git LFS
   git lfs install
   
   # Track model files
   git lfs track "*.safetensors"
   git lfs track "*.bin"
   git lfs track "outputs/grpo/checkpoints/**/*.safetensors"
   
   # Add and commit
   git add .gitattributes
   git commit -m "chore: add Git LFS tracking for model files"
   ```

2. **Then commit checkpoints**:
   ```bash
   git add outputs/grpo/checkpoints/...
   git commit -m "checkpoint: trained model epoch 5"
   git push
   ```

### Option 2: External Storage (Recommended)

Better for large models:
- **Shared network drive** (if on same network)
- **Cloud storage** (Google Drive, Dropbox, S3)
- **Model registry** (Hugging Face Hub, Weights & Biases Artifacts)
- **Shared server** (if team has one)

### Option 3: WandB Artifacts

Since you're already using WandB:

```python
# In your training script or after training
import wandb

# Log model as artifact
artifact = wandb.Artifact('gsm8k-model', type='model')
artifact.add_dir('./outputs/grpo/checkpoints/...')
wandb.log_artifact(artifact)
```

Team members can download:
```python
import wandb

run = wandb.Api().run("your-project/gsm8k-grpo-docker_trial0_train")
artifact = run.use_artifact('gsm8k-model:latest')
artifact.download('./downloaded_model')
```

## Best Practices for Team Collaboration

### 1. Configuration Files

- ✅ Commit: `gsm8k_grpo.yaml` (base config)
- ❌ Don't commit: `wandb/.wandb_api_key` (personal secrets)
- ✅ Use environment variables for secrets

### 2. Experiment Tracking

- ✅ All use same WandB project: `gsm8k-grpo-local`
- ✅ Use different `trial_name` for each team member
- ✅ Example: `trial_name=tongz_trial0`, `trial_name=alice_trial0`

### 3. Code Changes

- ✅ Keep `examples/math/` untouched (original AReaL code)
- ✅ Put Docker-specific changes in `examples/docker_gsm8k/`
- ✅ Document changes in commit messages

### 4. Docker Image Versioning

- ✅ Pin Docker image version: `ghcr.io/inclusionai/areal-runtime:v0.3.4`
- ✅ Document if team needs to update image version
- ✅ Consider building custom image if making AReaL changes

## Quick Reference Commands

### Starting Container (for team)

```bash
# Template - each person adjusts their path
docker run -it --name areal-grpo \
    --gpus all \
    --ipc=host \
    --shm-size=16g \
    -v /mnt/c/Users/USERNAME/path/to/AReaL:/workspace/AReaL:rw \
    -w /workspace/AReaL \
    ghcr.io/inclusionai/areal-runtime:v0.3.4 \
    /bin/bash
```

### Git from Container

```bash
# Set identity
git config user.name "Your Name"
git config user.email "your@email.com"

# Normal git workflow
git status
git add .
git commit -m "message"
git push
```

### Sharing Setup

1. Share this file: `examples/docker_gsm8k/TEAM_COLLABORATION.md`
2. Share Docker setup docs: `examples/local_gsm8k/WINDOWS_DOCKER_SETUP.md`
3. Team members follow steps above

## Troubleshooting

### Git not working in container?

1. Check volume mount:
   ```bash
   ls -la /workspace/AReaL/.git
   # Should show .git directory
   ```

2. Check git config:
   ```bash
   git config --list
   ```

3. Mount with SSH (if needed):
   ```bash
   docker run ... \
       -v ~/.ssh:/root/.ssh:ro \
       ...
   ```

### Different Windows paths?

Team members on different machines will have different paths:
- Adjust the `-v` flag in docker run command
- Use absolute paths: `/mnt/c/Users/USERNAME/...`

### GPU not accessible?

- Check NVIDIA drivers are up to date
- Verify Docker can see GPU: `docker run --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi`

---

## Summary

✅ **Sharing**: Team clones repo, follows Docker setup steps, uses shared configs  
✅ **Git from Docker**: Works! Set git config, then commit/push normally  
✅ **Same filesystem**: Changes in container = changes on host  
✅ **Best practice**: Edit on host, run in Docker, commit from either

