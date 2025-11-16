# Docker GSM8K GRPO Training

Docker-based GRPO training setup for single-GPU environments (e.g., Windows 11 with NVIDIA GPU, macOS, Linux).

## Features

- **Single-GPU Support**: Automatically uses disk-based weight updates instead of NCCL for single-GPU setups
- **Docker-Ready**: Configured for running in Docker containers with GPU access
- **GSM8K Dataset**: Trains on the GSM8K math problem dataset
- **Qwen 0.5B Model**: Uses Qwen2.5-0.5B-Instruct for fast training on consumer GPUs
- **Multiple Training Modes**: Fast (20-30 min), 1-hour, 3-hour, and full dataset training

## Quick Start

### Prerequisites

- Docker Desktop installed (with WSL2 integration for Windows)
- NVIDIA Container Toolkit (for GPU support)
- NVIDIA GPU with CUDA support

### Enter Docker Container

#### Option 1: If Container Already Running

```bash
docker exec -it areal-grpo bash
```

#### Option 2: Start New Container

```bash
# In WSL2 or PowerShell
cd /path/to/AReaL

docker run -it --name areal-grpo \
    --gpus all \
    --ipc=host \
    --shm-size=16g \
    -v /path/to/AReaL:/workspace/AReaL:rw \
    -w /workspace/AReaL \
    ghcr.io/inclusionai/areal-runtime:v0.3.4 \
    /bin/bash
```

**Note**: If container name already exists, remove it first:
```bash
docker stop areal-grpo && docker rm areal-grpo
```

### Run Training

Inside the container:

```bash
cd /workspace/AReaL

# Set WandB API key (optional)
export WANDB_API_KEY=your-api-key-here

# Fast training (20-30 minutes, 200 samples)
python3 -m areal.launcher.local examples/docker_gsm8k/gsm8k_grpo_train.py \
    --config examples/docker_gsm8k/gsm8k_grpo_fast.yaml \
    experiment_name=gsm8k-grpo-docker-fast \
    trial_name=trial0

# 1-hour training (500 samples, 2 epochs)
python3 -m areal.launcher.local examples/docker_gsm8k/gsm8k_grpo_train.py \
    --config examples/docker_gsm8k/gsm8k_grpo_1hour.yaml \
    experiment_name=gsm8k-grpo-docker-1hour \
    trial_name=trial0

# 3-hour training (1000 samples, 3 epochs)
python3 -m areal.launcher.local examples/docker_gsm8k/gsm8k_grpo_train.py \
    --config examples/docker_gsm8k/gsm8k_grpo_3hour.yaml \
    experiment_name=gsm8k-grpo-docker-3hour \
    trial_name=trial0

# Full dataset training (all samples, 5 epochs, ~5 days)
python3 -m areal.launcher.local examples/docker_gsm8k/gsm8k_grpo_train.py \
    --config examples/docker_gsm8k/gsm8k_grpo_full.yaml \
    experiment_name=gsm8k-grpo-docker-full \
    trial_name=trial0
```

## Files

- `gsm8k_grpo_train.py` - **Consolidated training script** (handles all configurations)
  - Handles all training configurations (fast, 1hour, 3hour, full)
  - Configuration is controlled via YAML files and command-line overrides
- `gsm8k_grpo.yaml` - Base configuration (full dataset)
- `gsm8k_grpo_fast.yaml` - Fast training (20-30 min, 200 samples)
- `gsm8k_grpo_1hour.yaml` - 1-hour training (500 samples)
- `gsm8k_grpo_3hour.yaml` - 3-hour training (1000 samples)
- `gsm8k_grpo_full.yaml` - Full dataset training (all samples)
- `run_training.sh` - Training launcher script
- `run_full_training.sh` - Multi-session full dataset training script
- `test_trained_model.py` - Model evaluation script
- `README.md` - This file
- `TRAINING_LEARNINGS.md` - Consolidated learnings and best practices

## Key Features

### Single-GPU Weight Updates

The training script automatically detects single-GPU setups and uses disk-based weight updates:

```python
# Automatically use disk-based updates for single GPU
if config.cluster.n_gpus_per_node == 1 and allocation_mode.gen.world_size == 1:
    weight_update_meta = WeightUpdateMeta.from_disk(...)
else:
    weight_update_meta = WeightUpdateMeta.from_fsdp_xccl(allocation_mode)
```

This avoids the NCCL "duplicate GPU" error that occurs when training and inference processes share the same GPU.

### Automatic Checkpoint Recovery

All training scripts support automatic recovery from checkpoints:

```bash
# Training automatically resumes from last checkpoint if available
# No manual intervention needed
```

### WandB Integration

Training progress is automatically logged to WandB:
- Project: `gsm8k-grpo-local`
- Metrics: Task reward (accuracy), loss, entropy, gradient norm
- View at: https://wandb.ai

## Configuration

Key settings for single-GPU Docker training:

- `cluster.n_gpus_per_node: 1` - Single GPU
- `allocation_mode: sglang.d1p1t1+d1p1t1` - Single GPU allocation
- `actor.path: Qwen/Qwen2.5-0.5B-Instruct` - Small model for single GPU
- `train_dataset.batch_size: 8` - Adjusted for single GPU memory

## Testing Trained Models

```bash
# Test a trained model
python examples/docker_gsm8k/test_trained_model.py \
    --model-path ./outputs/grpo/checkpoints/root/gsm8k-grpo-docker/trial0/default/epoch0epochstep0globalstep1 \
    --max-samples 10
```

## Viewing Training Progress

### WandB Dashboard

Visit: https://wandb.ai → Your project → `gsm8k-grpo-local`

Key metrics:
- `grpo_actor/task_reward/avg` - Accuracy (0.0 = 0%, 1.0 = 100%)
- `grpo_actor/loss` - Training loss
- `grpo_actor/entropy` - Policy entropy

### Training Logs

```bash
# View logs from outside container
docker logs -f areal-grpo

# Or from inside container
tail -f /workspace/AReaL/outputs/grpo/logs/root/gsm8k-grpo-docker/trial0/trainer.log
```

## Troubleshooting

See `TRAINING_LEARNINGS.md` for detailed troubleshooting, common issues, and best practices.

Common issues:
- **NCCL errors**: Single-GPU fix handles this automatically
- **Out of memory**: Reduce batch size or use faster training config
- **Checkpoints not saving**: Verify paths in config file

## Learn More

- **Training Learnings**: See `TRAINING_LEARNINGS.md` for detailed guides, GRPO explanation, and best practices
