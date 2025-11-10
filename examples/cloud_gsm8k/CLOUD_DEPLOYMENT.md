# Cloud Deployment Guide for AReaL GRPO Training

This guide helps you deploy AReaL GRPO training on cloud GPU platforms like Lambda AI, RunPod, Vast.ai, and similar services.

## Overview

Cloud platforms provide:
- ✅ High-end GPUs (A100, H100, etc.)
- ✅ Pre-configured Docker environments
- ✅ Persistent storage for checkpoints
- ✅ Pay-per-use pricing

## Prerequisites

1. **Account on cloud platform** (Lambda AI, RunPod, Vast.ai, etc.)
2. **WandB API key** (for experiment tracking)
3. **GitHub access** (to clone AReaL repository)

## General Cloud Setup Strategy

### Option 1: Clone Repository Inside Container (Recommended)

**Pros:**
- Simple setup
- Always uses latest code
- No volume mounting needed

**Cons:**
- Need to clone on each run (or use persistent storage)

### Option 2: Mount Cloud Storage Volume

**Pros:**
- Persistent code and checkpoints
- Faster startup

**Cons:**
- Need to set up cloud storage
- More complex setup

## Platform-Specific Guides

### Lambda AI

See `lambda_ai_setup.md` for detailed Lambda AI instructions.

**Quick Start:**
```bash
# Use the provided Docker run command
bash examples/cloud_gsm8k/docker_run_cloud.sh
```

### RunPod

See `runpod_setup.md` for detailed RunPod instructions.

**Quick Start:**
- Use RunPod's template system
- Or use the Docker run command in their console

### Vast.ai

See `vast_ai_setup.md` for detailed Vast.ai instructions.

**Quick Start:**
- Select a GPU instance
- Use the provided Docker run command in SSH

## Common Docker Run Command

All platforms support this basic format:

```bash
docker run -it --gpus all \
    --ipc=host \
    --shm-size=16g \
    -e WANDB_API_KEY=your-api-key-here \
    -e PYTHONPATH=/workspace/AReaL \
    -v /path/to/persistent/storage:/workspace/outputs \
    ghcr.io/inclusionai/areal-runtime:v0.3.4 \
    /bin/bash
```

## Training Scripts

Once inside the container, use:

```bash
# Clone repository (if not mounted)
git clone -b DL4Math https://github.com/nexthybrid/AReaL.git /workspace/AReaL
cd /workspace/AReaL

# Install AReaL
pip install -e .

# Run training
bash examples/cloud_gsm8k/run_training_cloud.sh
```

## Configuration Files

- `gsm8k_grpo_cloud.yaml` - Cloud-optimized configuration
- Adjusts paths for cloud environments
- Optimized for cloud GPU instances

## Persistent Storage

### What to Save

1. **Checkpoints**: `outputs/grpo/checkpoints/`
2. **Logs**: `outputs/grpo/logs/`
3. **WandB logs**: Synced automatically if WandB API key is set
4. **Model checkpoints**: For inference later

### Storage Recommendations

- **Lambda AI**: Use their persistent storage volumes
- **RunPod**: Use their network volumes
- **Vast.ai**: Mount external storage or use their volume system

## Environment Variables

Set these in your cloud platform:

```bash
WANDB_API_KEY=your-api-key-here
PYTHONPATH=/workspace/AReaL
CUDA_VISIBLE_DEVICES=0  # If multiple GPUs
```

## Monitoring

### WandB Dashboard

All training metrics are logged to WandB:
- Project: `gsm8k-grpo-local`
- View at: https://wandb.ai

### Container Logs

Most platforms provide log viewing:
- Lambda AI: Web console
- RunPod: Logs tab
- Vast.ai: SSH access

## Troubleshooting

### GPU Not Detected

```bash
# Inside container
nvidia-smi
```

If not working, check platform GPU passthrough settings.

### Out of Memory

- Reduce `batch_size` in config
- Reduce `max_new_tokens`
- Use gradient checkpointing

### Network Issues

- Some platforms need `--network host`
- Check firewall settings
- Verify SGLang server can bind to ports

## Cost Optimization

1. **Use spot instances** (if available) - 50-70% cheaper
2. **Monitor GPU utilization** - Ensure GPU is being used
3. **Stop when done** - Don't leave instances running
4. **Use appropriate GPU** - A100 for training, smaller GPUs for testing

## Next Steps

1. Choose your platform
2. Follow platform-specific guide
3. Run training script
4. Monitor in WandB

