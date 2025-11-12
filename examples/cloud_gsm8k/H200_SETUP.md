# H200 GPU Full Dataset Training Setup

## Overview

H200 GPU (141GB memory) is ideal for full dataset GRPO training. This setup uses the complete GSM8K dataset with 5 epochs.

## Quick Start

```bash
bash examples/cloud_gsm8k/run_training_cloud.sh h200
```

## Configuration

The `h200` config option uses:
- **Config**: `gsm8k_grpo_cloud.yaml`
- **Script**: `gsm8k_grpo_cloud.py`
- **Dataset**: Full GSM8K dataset (no limiting)
- **Epochs**: 5
- **Experiment name**: `gsm8k-grpo-cloud-h200`

## Key Settings

- **Batch size**: 8
- **Max concurrent rollouts**: 32
- **Max new tokens**: 512
- **Memory fraction (SGLang)**: 0.8 (80% of 141GB = ~113GB for SGLang)
- **Gradient checkpointing**: Disabled (H200 has plenty of memory)

## Expected Performance

- **Training time**: ~5 days (full dataset, 5 epochs)
- **Memory usage**: ~40-50GB total (comfortable for H200's 141GB)
- **Throughput**: Higher than smaller GPUs due to larger batch sizes and more concurrent rollouts

## H200 Advantages

- **141GB memory**: Can handle full dataset without memory optimizations
- **No gradient checkpointing needed**: Faster training
- **Larger batch sizes**: Better training stability
- **More concurrent rollouts**: Higher inference throughput

## Attention Backend

H200 (SM 90) works with the default `fa3` (FlashAttention v3) backend, so no special configuration needed.

## Monitoring

Check training progress:
- **WandB**: https://wandb.ai (project: `gsm8k-grpo-local`)
- **Logs**: `/workspace/outputs/grpo/logs/root/gsm8k-grpo-cloud-h200/`
- **Checkpoints**: `/workspace/outputs/grpo/checkpoints/gsm8k-grpo-cloud-h200/`

## Comparison with Other GPUs

| GPU | Memory | Config | Dataset | Epochs | Time |
|-----|--------|--------|---------|--------|------|
| RTX 4090 | 24GB | 1hour/3hour | Limited | 2-3 | 1-3 hours |
| A40 | 44GB | 1hour_a40/3hour_a40 | Limited | 2-3 | 1-4.5 hours |
| A100 | 40/80GB | full | Full | 5 | ~5 days |
| H200 | 141GB | h200 | Full | 5 | ~5 days |

## Troubleshooting

If you encounter issues:

1. **Check GPU**: `nvidia-smi` should show H200 with 141GB
2. **Check memory usage**: Should stay well under 141GB
3. **Check logs**: Look for errors in `/workspace/outputs/grpo/logs/`

## Future Optimizations

With H200's large memory, you could potentially:
- Increase batch size to 16 or 32
- Increase max_concurrent_rollouts to 64
- Increase max_new_tokens if needed
- Disable gradient checkpointing (already done)

These optimizations would require creating an H200-specific config file.

