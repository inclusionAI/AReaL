# Cloud Setup Summary

## Files Created

All cloud deployment files are in `examples/cloud_gsm8k/`:

### Documentation
- `README.md` - Overview and quick reference
- `CLOUD_DEPLOYMENT.md` - Comprehensive cloud deployment guide
- `QUICK_START_CLOUD.md` - Quick start guide
- `lambda_ai_setup.md` - Lambda AI specific instructions
- `runpod_setup.md` - RunPod specific instructions
- `vast_ai_setup.md` - Vast.ai specific instructions

### Scripts
- `docker_run_cloud.sh` - Docker run script for cloud platforms
- `run_training_cloud.sh` - Training script with config selection
- `test_trained_model_cloud.py` - Evaluation script for cloud

### Configuration Files
- `gsm8k_grpo_cloud.yaml` - Full training config (cloud-optimized)
- `gsm8k_grpo_1hour.yaml` - 1-hour training config
- `gsm8k_grpo_3hour.yaml` - 3-hour training config
- `gsm8k_grpo_fast.yaml` - Fast training config (20-30 min)

## Quick Start Commands

### 1. Set Up Environment

```bash
# Set WandB API key
export WANDB_API_KEY=your-api-key-here

# Set project path (adjust for your platform)
export PROJECT_PATH=/workspace/AReaL
export OUTPUTS_PATH=/workspace/outputs
```

### 2. Run Docker Container

```bash
# Use provided script
bash examples/cloud_gsm8k/docker_run_cloud.sh

# Or manually
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

### 3. Set Up Inside Container

```bash
# Inside container
cd /workspace
git clone -b DL4Math https://github.com/nexthybrid/AReaL.git
cd AReaL
pip install -e .
```

### 4. Run Training

```bash
# Inside container
cd /workspace/AReaL

# Fast training (20-30 min)
bash examples/cloud_gsm8k/run_training_cloud.sh fast

# 1-hour training
bash examples/cloud_gsm8k/run_training_cloud.sh 1hour

# 3-hour training
bash examples/cloud_gsm8k/run_training_cloud.sh 3hour

# Full training (takes days)
bash examples/cloud_gsm8k/run_training_cloud.sh full
```

### 5. Evaluate Model

```bash
# Inside container, after training
python3 -m areal.launcher.local examples/cloud_gsm8k/test_trained_model_cloud.py \
    --config examples/cloud_gsm8k/gsm8k_grpo_1hour.yaml \
    experiment_name=gsm8k-grpo-cloud-eval \
    trial_name=full_test \
    rollout.experiment_name=gsm8k-grpo-cloud-1hour \
    rollout.trial_name=trial0
```

## Platform Comparison

| Platform | Pros | Cons | Best For |
|----------|------|------|----------|
| **Lambda AI** | Easy setup, reliable | Higher cost | Beginners, production |
| **RunPod** | Good pricing, spot instances | More setup | Cost-conscious users |
| **Vast.ai** | Cheapest | More manual setup | Budget users |

## Key Differences from Local

1. **No local mounts**: Code cloned inside container
2. **Persistent storage**: Use cloud volumes for checkpoints
3. **Network**: May need `--network host`
4. **Environment**: WandB API key via environment variable

## Cost Estimates

For 3-hour training:
- **Lambda AI A100**: ~$3.30
- **RunPod RTX 4090**: ~$0.87
- **RunPod A100 (Spot)**: ~$1.25-2.08
- **Vast.ai RTX 4090**: ~$0.60-1.20

## Next Steps

1. Choose your platform
2. Follow platform-specific guide
3. Run training script
4. Monitor in WandB
5. Download results before stopping instance

See individual platform guides for detailed instructions!

