# Local GSM8K Training for Mac M2

This directory contains a simplified training setup for fine-tuning models on GSM8K dataset on a local Mac M2 machine.

## Overview

This training script bypasses the complex distributed infrastructure of AReaL and provides a simple, standalone training setup that works on local hardware (CPU/MPS for Mac).

### Key Features

- ✅ Simple standalone training (no distributed setup needed)
- ✅ Mac M2 support (CPU and MPS backends)
- ✅ Automatic device selection (CPU/MPS/CUDA)
- ✅ Memory-efficient training with gradient checkpointing
- ✅ W&B integration for experiment tracking
- ✅ Time-limited training for quick test runs
- ✅ Uses AReaL's dataset utilities and reward functions

## Requirements

```bash
# Activate virtual environment
source venv/bin/activate

# Install required packages
pip install torch transformers datasets wandb accelerate tqdm
```

## Quick Start

### Basic Training (30-minute test run)

```bash
python examples/local_gsm8k/train_local_simple.py \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --output-dir ./outputs/gsm8k-test \
    --max-time 1800 \
    --batch-size 2 \
    --gradient-accumulation-steps 16 \
    --max-epochs 3
```

### Custom Configuration

```bash
python examples/local_gsm8k/train_local_simple.py \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --output-dir ./outputs/gsm8k-local \
    --lr 5e-5 \
    --batch-size 4 \
    --max-epochs 5 \
    --max-time 3600 \
    --wandb \
    --wandb-project my-gsm8k-experiment
```

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model` | Model path or HuggingFace ID | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` |
| `--output-dir` | Output directory | `./outputs/gsm8k-local` |
| `--lr` | Learning rate | `5e-5` |
| `--batch-size` | Batch size | `4` (use 2 for 30-min test) |
| `--max-epochs` | Maximum epochs | `3` |
| `--max-steps` | Maximum training steps | `None` |
| `--max-time` | Maximum time in seconds | `1800` (30 min) |
| `--gradient-accumulation-steps` | Gradient accumulation | `8` (use 16 for 30-min test) |
| `--max-length` | Max sequence length | `512` |
| `--save-steps` | Save checkpoint every N steps | `50` |
| `--wandb` | Use W&B tracking | `True` |
| `--wandb-project` | W&B project name | `areal-gsm8k-mac` |
| `--device` | Device to use | `auto` (detects MPS/CUDA/CPU) |

## Memory Optimization Tips

For 32GB RAM Mac M2:

1. **Small batch size**: Use `--batch-size 2`
2. **Large gradient accumulation**: Use `--gradient-accumulation-steps 16`
3. **Limit dataset**: The script automatically limits to 500 samples for quick tests
4. **Gradient checkpointing**: Automatically enabled for memory efficiency

## Output

- **Model checkpoints**: Saved in `output-dir` (latest) and `output-dir/checkpoint-*` (intermediate)
- **W&B logs**: Automatically logged to your W&B project
- **Console logs**: Training progress and metrics

## Example Run

```bash
$ python examples/local_gsm8k/train_local_simple.py --max-time 1800 --batch-size 2

Loading model from deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
Using MPS (Metal Performance Shaders) backend
Device: mps
Loading GSM8K dataset...
Training on 500 samples
Starting training for 3 epochs
Total steps: 188
Steps per epoch: 500
Epoch 1/3: 100%|████████████████████| 500/500 [10:23<00:00, 1.25s/it, loss=2.345]
Epoch 1 average loss: 2.345
Saved checkpoint to ./outputs/gsm8k-local/checkpoint-step_50
...
Training completed!
```

## Troubleshooting

### Out of Memory

- Reduce batch size: `--batch-size 1`
- Increase gradient accumulation: `--gradient-accumulation-steps 32`
- Reduce max length: `--max-length 256`

### Slow Training on CPU

- Training on CPU is slow. Use MPS (Metal) backend if available on your Mac
- For faster training, consider using a cloud GPU or reducing dataset size

### Model Issues

If you encounter issues with the model:

```bash
# Try a smaller model
--model Qwen/Qwen2.5-1.5B-Instruct
```

## Next Steps

1. **Compare before/after**: Test the model on GSM8K examples
2. **Extend training**: Run longer with `--max-time 7200` (2 hours)
3. **Evaluate**: Use the AReaL evaluation utilities
4. **Submit to leaderboard**: Check the model on HuggingFace leaderboard

## Integration with AReaL

This script is designed to be complementary to the main AReaL framework:

- **For local development and testing**: Use this script
- **For production training**: Use the main AReaL framework with GPU clusters
- **Dataset compatibility**: Uses same AReaL dataset utilities
- **Model compatibility**: Saved checkpoints can be loaded by AReaL

