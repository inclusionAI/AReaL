# Quick Start: Cloud Deployment

## ⭐ Recommended: RunPod

**RunPod is the most economical option!** See `RUNPOD_QUICK_START.md` for RunPod-specific quick start.

## Overview

This guide helps you quickly deploy AReaL GRPO training on cloud GPU platforms.

## Step 1: Choose Your Platform

- **Lambda AI**: https://lambdalabs.com (Recommended for beginners)
- **RunPod**: https://runpod.io (Good pricing, spot instances)
- **Vast.ai**: https://vast.ai (Cheapest, but more setup)

## Step 2: Set Up Platform

Follow the platform-specific guide:
- `lambda_ai_setup.md` - Lambda AI
- `runpod_setup.md` - RunPod  
- `vast_ai_setup.md` - Vast.ai

## Step 3: Run Docker Container

### Option A: Use Provided Script

```bash
# Set WandB API key
export WANDB_API_KEY=your-api-key-here

# Run container
bash examples/cloud_gsm8k/docker_run_cloud.sh
```

### Option B: Manual Docker Run

```bash
# Set WandB API key
export WANDB_API_KEY=your-api-key-here

# Run container
docker run -it --name areal-grpo-cloud \
    --gpus all \
    --ipc=host \
    --shm-size=16g \
    --network host \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    -e PYTHONPATH=/workspace/AReaL \
    -v /workspace/AReaL:/workspace/AReaL:rw \
    -v /workspace/outputs:/workspace/outputs:rw \
    -w /workspace/AReaL \
    ghcr.io/inclusionai/areal-runtime:v0.3.4 \
    /bin/bash
```

## Step 4: Set Up Inside Container

```bash
# Inside container
cd /workspace

# Clone repository (if not mounted)
git clone -b DL4Math https://github.com/nexthybrid/AReaL.git
cd AReaL

# Install AReaL
pip install -e .

# Verify GPU
nvidia-smi
```

## Step 5: Run Training

```bash
# Inside container
cd /workspace/AReaL

# Set WandB API key (if not set via environment)
export WANDB_API_KEY=your-api-key-here

# Run training (choose one):
# Fast training (20-30 min)
bash examples/cloud_gsm8k/run_training_cloud.sh fast

# 1-hour training
bash examples/cloud_gsm8k/run_training_cloud.sh 1hour

# 3-hour training
bash examples/cloud_gsm8k/run_training_cloud.sh 3hour

# Full training (takes days)
bash examples/cloud_gsm8k/run_training_cloud.sh full
```

## Step 6: Monitor Training

1. **WandB Dashboard**: https://wandb.ai
   - Project: `gsm8k-grpo-local`
   - View training curves and metrics

2. **Container Logs**: Check platform's log viewer

3. **Checkpoints**: Saved to `outputs/grpo/checkpoints/`

## Step 7: Download Results

Before stopping the instance:

```bash
# Download checkpoints (from host, outside container)
# Adjust paths based on your platform
scp -r user@instance-ip:/workspace/outputs ./local_outputs
```

Or use cloud storage (S3, GCS):

```bash
# Inside container
aws s3 sync /workspace/outputs s3://your-bucket/areal-outputs/
```

## Platform-Specific Quick Commands

### Lambda AI

```bash
# SSH to instance
ssh ubuntu@<instance-ip>

# Then follow Step 3-5 above
```

### RunPod

```bash
# Use RunPod web terminal or Jupyter
# Then follow Step 3-5 above
```

### Vast.ai

```bash
# SSH to instance
ssh root@<instance-ip> -p <port>

# Then follow Step 3-5 above
```

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
- Enable `gradient_checkpointing: true`

### Network Issues

- Use `--network host` in Docker run
- Check platform firewall settings
- Verify SGLang server can bind to ports

## Cost Tips

1. **Use spot instances** (if available) - 50-70% cheaper
2. **Stop when done** - Don't leave instances running
3. **Monitor GPU utilization** - Ensure GPU is being used
4. **Use appropriate GPU** - A100 for training, smaller for testing

## Next Steps

- See `CLOUD_DEPLOYMENT.md` for detailed guide
- See platform-specific guides for advanced setup
- Check `HOW_TO_VIEW_TRAINING_CURVES.md` for monitoring

