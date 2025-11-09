# Running GRPO Training - Fixed Config

## Fixed Issues

1. ✅ **WandB API Key**: Changed from `${env.WANDB_API_KEY}` to `${oc.env:WANDB_API_KEY}` (OmegaConf format)
2. ✅ **WandB API Key stored**: Saved in `wandb/.wandb_api_key`

## Current Status

Training starts successfully but fails during weight update broadcast with NCCL error on single GPU.

## Running Training

```bash
# Inside Docker container
cd /workspace/AReaL
export WANDB_API_KEY=$(cat wandb/.wandb_api_key 2>/dev/null || echo "e1adc5be02c03fd34828c84b1ece937e0c2feb6e")

python3 -m areal.launcher.local examples/docker_gsm8k/gsm8k_grpo.py \
    --config examples/docker_gsm8k/gsm8k_grpo.yaml \
    experiment_name=gsm8k-grpo-docker \
    trial_name=trial0
```

## What Works

- Config loading ✅
- Model download ✅  
- Dataset loading ✅
- SGLang server startup ✅
- First training step ✅
- Rollout generation ✅
- Reward computation ✅
- PPO update ✅

## Known Issue

- ❌ Weight update broadcast fails on single GPU (NCCL duplicate GPU error)

This needs further investigation or AReaL single-GPU mode configuration.

