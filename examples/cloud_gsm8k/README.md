# Cloud Deployment Guide for GRPO Training

This directory contains scripts and configurations for running AReaL GRPO training on cloud GPU platforms, **optimized for RunPod** (most economical option).

## Recommended Platform: RunPod

**Why RunPod?**
- 💰 **Best pricing**: RTX 4090 at $0.29/hour (spot: $0.09/hour!)
- 📦 **Network volumes**: Persistent storage for checkpoints
- 🚀 **Easy setup**: Template system
- ⚡ **Fast startup**: Pods ready in seconds

**Quick Start**: See `RUNPOD_QUICK_START.md` or `RUNPOD_COMPLETE_GUIDE.md`

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

**💡 Important**: Set your WandB API key as an environment variable in RunPod (see `RUNPOD_COMPLETE_GUIDE.md` for details)

## Files

### Main Training Script
- `run_training_cloud.sh` - **Main training script** - Use this to start training
  - Supports: `fast`, `1hour`, `3hour`, `full`, `h200` configs
  - Auto-detects GPU type (A40, H200, etc.) and uses appropriate config
  - Usage: `bash examples/cloud_gsm8k/run_training_cloud.sh [config_name]`

### RunPod Documentation
- `RUNPOD_QUICK_START.md` - ⭐ **Start here for RunPod** - Quick setup guide
- `RUNPOD_COMPLETE_GUIDE.md` - Complete RunPod guide with troubleshooting
- `runpod_template.json` - RunPod template configuration (optional)

### Training Configurations
- `gsm8k_grpo_cloud.yaml` - Full training (cloud-optimized, for H200/full runs)
- `gsm8k_grpo_1hour.yaml` - 1-hour training (RTX 4090/A100)
- `gsm8k_grpo_1hour_a40.yaml` - 1-hour training (A40 GPU optimized) ⭐
- `gsm8k_grpo_3hour.yaml` - 3-hour training
- `gsm8k_grpo_3hour_a40.yaml` - 3-hour training (A40 GPU optimized)
- `gsm8k_grpo_fast.yaml` - Fast training (20-30 min)

### Training Scripts
- `gsm8k_grpo_cloud.py` - Main training script (used by `run_training_cloud.sh`)

### Recovery and Checkpoint Management
- `list_checkpoints.py` - List available checkpoints
- `setup_recovery.py` - Set up recovery from a checkpoint
- `resume_training.py` - Interactive recovery guide
- `CHECKPOINT_AND_RECOVERY_GUIDE.md` - Detailed recovery documentation
- `RECOVERY_QUICK_START.md` - Quick recovery reference
- `CIRCUIT_BREAKER_AND_RECOVERY_SUMMARY.md` - Circuit breaker implementation summary

### GPU-Specific Documentation
- `H200_SETUP.md` - H200 GPU setup and configuration
- `H200_STEP188_DIAGNOSIS.md` - Analysis of H200 training crash at step 188
- `A40_GPU_FIX.md` - A40 GPU memory optimization guide
- `A40_3HOUR_SUMMARY.md` - A40 3-hour training summary
- `1HOUR_VS_3HOUR_COMPARISON.md` - Comparison of 1-hour vs 3-hour configs

### Other Documentation
- `CHECKPOINT_SAVING_FIX.md` - Checkpoint saving configuration fixes
- `test_trained_model_cloud.py` - Model evaluation script

## Cost Comparison

For 3-hour training:
- **RunPod RTX 4090 Spot**: ~$0.27 ⭐ **Best Value**
- **RunPod RTX 4090**: ~$0.87
- **RunPod A40**: ~$1.20-1.50 (if RTX 4090 unavailable)
- **RunPod H200**: ~$4.50-6.00 (for full dataset training)

**Note**: A40 GPU requires memory-optimized config. The training script auto-detects A40 and uses `gsm8k_grpo_1hour_a40.yaml` or `gsm8k_grpo_3hour_a40.yaml` automatically.

## Key Features

1. **Automatic GPU Detection**: Script detects GPU type and uses appropriate config
2. **Circuit Breaker**: Training stops automatically if task reward is zero for 10 consecutive steps
3. **Checkpoint Recovery**: Easy recovery from checkpoints using provided scripts
4. **Network Volumes**: Persistent storage for checkpoints across pod restarts
5. **Spot Instances**: Enable for 50-70% cost savings

## Important: Using Your Forked Branch

**This setup uses your forked repository**: `nexthybrid/AReaL` branch `DL4Math`

- ✅ **Docker Image**: Uses `ghcr.io/inclusionai/areal-runtime:v0.3.4` (provides runtime environment)
- ✅ **Code**: Clones from `https://github.com/nexthybrid/AReaL` branch `DL4Math`
- ✅ **Your custom scripts**: Uses your modified `examples/cloud_gsm8k/` scripts

**Why the original Docker image works**: The image only provides the runtime environment (CUDA, PyTorch, dependencies). Your actual code gets cloned from GitHub, so you can use your forked branch without building a new image.

## Troubleshooting

- **Pip installation fails**: See `RUNPOD_COMPLETE_GUIDE.md` - "Pip Installation Fails" section
- **CUDA Out of Memory on A40**: See `A40_GPU_FIX.md` ⚠️ or `RUNPOD_COMPLETE_GUIDE.md` - "CUDA Out of Memory" section
- **Checkpoints not saving**: See `CHECKPOINT_SAVING_FIX.md` ⚠️
- **Training crashes**: See `RUNPOD_COMPLETE_GUIDE.md` - "Training Crashes" section
- **Task reward drops to zero**: See `H200_STEP188_DIAGNOSIS.md` and `CIRCUIT_BREAKER_AND_RECOVERY_SUMMARY.md`
- **Resuming training**: See `CHECKPOINT_AND_RECOVERY_GUIDE.md` or `RECOVERY_QUICK_START.md`

## Next Steps

1. **For RunPod Setup**: See `RUNPOD_QUICK_START.md` ⭐
2. **For Complete Guide**: See `RUNPOD_COMPLETE_GUIDE.md`
3. **For A40 GPU Issues**: See `A40_GPU_FIX.md` ⚠️
4. **For H200 Setup**: See `H200_SETUP.md`
5. **For Recovery**: See `RECOVERY_QUICK_START.md`
