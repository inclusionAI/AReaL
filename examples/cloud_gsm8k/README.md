# Cloud Deployment Guide for GRPO Training

This directory contains scripts and configurations for running AReaL GRPO training on cloud GPU platforms, **optimized for RunPod** (most economical option).

## Recommended Platform: RunPod

**Why RunPod?**
- 💰 **Best pricing**: RTX 4090 at $0.29/hour (spot: $0.09/hour!)
- 📦 **Network volumes**: Persistent storage for checkpoints
- 🚀 **Easy setup**: Template system
- ⚡ **Fast startup**: Pods ready in seconds

**Quick Start**: See `RUNPOD_QUICK_START.md` or `RUNPOD_COMPLETE_GUIDE.md`

## Supported Platforms

- **RunPod** (https://runpod.io) ⭐ **Recommended - Most Economical**
- **Lambda AI** (https://lambdalabs.com) - Easiest setup
- **Vast.ai** (https://vast.ai) - Cheapest but more setup
- **Any platform with Docker + GPU support**

## Quick Start (RunPod)

### 1. Create RunPod Account
- Go to https://runpod.io
- Sign up and add credits

### 2. Create Network Volume
- Go to "Volumes" → "Create Volume"
- Name: `areal-outputs`, Size: 50GB

### 3. Deploy Pod
- Use template (see `RUNPOD_QUICK_START.md`)
- Or manual deployment (see `RUNPOD_COMPLETE_GUIDE.md`)

### 4. Run Training
```bash
# Inside pod
bash examples/cloud_gsm8k/run_training_cloud.sh 1hour
```

**💡 Important**: Set your WandB API key as an environment variable in RunPod (see `RUNPOD_WANDB_API_KEY.md` for details)

## Files

### RunPod-Specific (Recommended)
- `RUNPOD_QUICK_START.md` - ⭐ **Start here for RunPod**
- `RUNPOD_COMPLETE_GUIDE.md` - Complete RunPod guide
- `runpod_deploy.sh` - RunPod deployment helper script
- `runpod_docker_run.sh` - RunPod-optimized Docker run script
- `runpod_template.json` - RunPod template configuration

### General Cloud
- `CLOUD_DEPLOYMENT.md` - Comprehensive cloud deployment guide
- `QUICK_START_CLOUD.md` - General quick start
- `DOCKER_COMMANDS.md` - Copy-paste Docker commands
- `PLATFORM_COMPARISON.md` - Platform comparison

### Platform-Specific
- `lambda_ai_setup.md` - Lambda AI instructions
- `runpod_setup.md` - RunPod instructions (detailed)
- `vast_ai_setup.md` - Vast.ai instructions

### Scripts
- `docker_run_cloud.sh` - Docker run script for cloud platforms
- `run_training_cloud.sh` - Training script with config selection
- `test_trained_model_cloud.py` - Evaluation script

### Configurations
- `gsm8k_grpo_cloud.yaml` - Full training (cloud-optimized)
- `gsm8k_grpo_1hour.yaml` - 1-hour training (RTX 4090/A100)
- `gsm8k_grpo_1hour_a40.yaml` - 1-hour training (A40 GPU optimized) ⭐
- `gsm8k_grpo_3hour.yaml` - 3-hour training
- `gsm8k_grpo_fast.yaml` - Fast training (20-30 min)

## Cost Comparison

For 3-hour training:
- **RunPod RTX 4090 Spot**: ~$0.27 ⭐ **Best Value**
- **RunPod RTX 4090**: ~$0.87
- **RunPod A40**: ~$1.20-1.50 (if RTX 4090 unavailable)
- **Lambda AI A100**: ~$3.30
- **Vast.ai RTX 4090**: ~$0.60-1.20

**Note**: A40 GPU requires memory-optimized config. The training script auto-detects A40 and uses `gsm8k_grpo_1hour_a40.yaml` automatically.

## Key Differences from Local Setup

1. **No local volume mounts**: Code is cloned inside container
2. **Persistent storage**: Use RunPod network volumes for checkpoints
3. **Network**: Uses `--network host` for SGLang server
4. **Environment**: WandB API key passed via environment variable
5. **Spot instances**: Enable for 50-70% cost savings

## Important: Using Your Forked Branch

**This setup uses your forked repository**: `nexthybrid/AReaL` branch `DL4Math`

- ✅ **Docker Image**: Still uses `ghcr.io/inclusionai/areal-runtime:v0.3.4` (original image is fine - see `DOCKER_IMAGE_EXPLANATION.md`)
- ✅ **Code**: Clones from `https://github.com/nexthybrid/AReaL` branch `DL4Math`
- ✅ **Your custom scripts**: Uses your modified `examples/cloud_gsm8k/` scripts

**Why the original Docker image works**: The image only provides the runtime environment (CUDA, PyTorch, dependencies). Your actual code gets cloned from GitHub, so you can use your forked branch without building a new image.

See `DOCKER_IMAGE_EXPLANATION.md` for detailed explanation.

## Troubleshooting

- **Pip installation fails**: See `RUNPOD_COMPLETE_GUIDE.md` - "Pip Installation Fails" section
- **CUDA Out of Memory on A40**: See `A40_GPU_FIX.md` ⚠️ or `RUNPOD_COMPLETE_GUIDE.md` - "CUDA Out of Memory" section
- **Container restart loop**: See `CONTAINER_RESTART_LOOP_FIX.md` ⚠️ (container restarts every ~17 seconds)
- **Training crashes around step 50-60**: See `CRASH_DIAGNOSIS.md` ⚠️ or `RUNPOD_COMPLETE_GUIDE.md` - "Training Crashes" section
- **Model persistence**: See `RUNPOD_COMPLETE_GUIDE.md` - "Resuming Training" section

## Next Steps

1. **For RunPod**: See `RUNPOD_QUICK_START.md` ⭐
2. **For A40 GPU issues**: See `A40_GPU_FIX.md` ⚠️
3. **For Docker image explanation**: See `DOCKER_IMAGE_EXPLANATION.md`
4. **For other platforms**: See `CLOUD_DEPLOYMENT.md`
5. **For comparison**: See `PLATFORM_COMPARISON.md`

