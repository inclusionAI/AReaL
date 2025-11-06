# Docker GSM8K GRPO Training

Docker-based GRPO training setup for single-GPU environments (e.g., Windows 11 with NVIDIA GPU).

## Features

- **Single-GPU Support**: Automatically uses disk-based weight updates instead of NCCL for single-GPU setups
- **Docker-Ready**: Configured for running in Docker containers with GPU access
- **GSM8K Dataset**: Trains on the GSM8K math problem dataset
- **Qwen 0.5B Model**: Uses Qwen2.5-0.5B-Instruct for fast training on consumer GPUs

## Files

- `gsm8k_grpo.py` - Training script with single-GPU fix (uses disk-based weight updates)
- `gsm8k_grpo.yaml` - Configuration file optimized for single-GPU Docker training
- `README.md` - This file

## Key Fix: Single-GPU Weight Updates

The training script automatically detects single-GPU setups and uses disk-based weight updates:

```python
# Automatically use disk-based updates for single GPU
if config.cluster.n_gpus_per_node == 1 and allocation_mode.gen.world_size == 1:
    weight_update_meta = WeightUpdateMeta.from_disk(...)
else:
    weight_update_meta = WeightUpdateMeta.from_fsdp_xccl(allocation_mode)
```

This avoids the NCCL "duplicate GPU" error that occurs when training and inference processes share the same GPU.

## Running Training

### Inside Docker Container

```bash
cd /workspace/AReaL
export WANDB_API_KEY=$(cat wandb/.wandb_api_key 2>/dev/null || echo "your-api-key")

python3 -m areal.launcher.local examples/docker_gsm8k/gsm8k_grpo.py \
    --config examples/docker_gsm8k/gsm8k_grpo.yaml \
    experiment_name=gsm8k-grpo-docker \
    trial_name=trial0
```

## Configuration

See `gsm8k_grpo.yaml` for all configuration options. Key settings for single-GPU:

- `cluster.n_gpus_per_node: 1` - Single GPU
- `allocation_mode: sglang.d1p1t1+d1p1t1` - Single GPU allocation
- `actor.path: Qwen/Qwen2.5-0.5B-Instruct` - Small model for single GPU
- `train_dataset.batch_size: 8` - Adjusted for single GPU memory

## Differences from Stock Example

1. **Single-GPU Detection**: Automatically switches to disk-based weight updates
2. **Smaller Model**: Uses 0.5B model instead of 1.5B
3. **Optimized Batch Size**: Reduced for single GPU memory constraints
4. **Docker Paths**: Config uses relative paths that work in Docker

## Troubleshooting

See documentation in `examples/local_gsm8k/`:
- `WINDOWS_DOCKER_SETUP.md` - Docker setup guide
- `SINGLE_GPU_FIX.md` - Explanation of the single-GPU fix
- `SINGLE_GPU_ANALYSIS.md` - Detailed analysis of the issue

## WandB API Key

Store your WandB API key in `wandb/.wandb_api_key` (not tracked by git):

```bash
mkdir -p wandb
echo "your-api-key" > wandb/.wandb_api_key
```

The training script will automatically load it if set as `WANDB_API_KEY` environment variable or read from the file.

