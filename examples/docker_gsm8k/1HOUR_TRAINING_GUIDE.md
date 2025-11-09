# 1-Hour Training Guide for GRPO

## Overview

This guide provides a **1-hour training configuration** that balances training time with meaningful dataset coverage. This is a middle ground between the fast training (20-30 min) and full training (5 days).

## Configuration Summary

| Setting | Fast (20-30 min) | 1-Hour | Full (5 days) |
|---------|----------------|--------|---------------|
| **Epochs** | 1 | 2 | 5 |
| **Dataset Size** | 200 samples | 500 samples | 7,473 samples |
| **Steps per Epoch** | ~25 | ~63 | ~934 |
| **Total Steps** | ~25 | ~126 | ~4,670 |
| **n_samples** | 2 | 4 | 4 |
| **max_new_tokens** | 256 | 512 | 512 |
| **Batch Size** | 8 | 8 | 8 |
| **Estimated Time** | ~20-30 min | ~1-2 hours | ~5 days |

## Key Parameters

### Dataset Size: 500 Samples
- **Rationale**: Provides good coverage of the training set while keeping training time reasonable
- **Steps per epoch**: 500 / 8 = ~63 steps
- **Total steps**: 2 epochs × 63 = ~126 steps

### Epochs: 2
- **Rationale**: Allows the model to see each sample twice, improving convergence
- Better than 1 epoch for learning, but faster than 5 epochs

### Generation Parameters
- **n_samples: 4**: Keeps full GRPO quality (4 solutions per problem)
- **max_new_tokens: 512**: Allows full reasoning chains

### Batch Size: 8
- Kept at 8 for stability and memory efficiency

## Files Created

1. **`gsm8k_grpo_1hour.yaml`** - 1-hour training configuration
2. **`gsm8k_grpo_1hour.py`** - Training script (limits dataset to 500 samples)

## Running 1-Hour Training

### Inside Docker Container

```bash
cd /workspace/AReaL

# Set WandB API key
export WANDB_API_KEY=$(cat examples/local_gsm8k/wandb/.wandb_api_key 2>/dev/null || echo "your-api-key")

# Run 1-hour training
python3 -m areal.launcher.local examples/docker_gsm8k/gsm8k_grpo_1hour.py \
    --config examples/docker_gsm8k/gsm8k_grpo_1hour.yaml \
    experiment_name=gsm8k-grpo-1hour \
    trial_name=trial0
```

### Expected Output

```
[1-HOUR TRAINING MODE] Limiting dataset from 7473 to 500 samples
[1-HOUR TRAINING MODE]
  Dataset size: 500 samples
  Batch size: 8
  Steps per epoch: 63
  Total epochs: 2
  Total steps: 126
  Estimated time: ~126 minutes (~2.1 hours) at ~1 step/min
```

**Note**: The actual time may be closer to 1-1.5 hours with optimizations and depending on your hardware.

## What You'll See

### Training Progress

- **~126 steps total** (vs 25 in fast, 4,670 in full)
- **~1 step per minute** (may vary based on hardware)
- **~1-2 hours total** training time
- **2 epochs** for better convergence

### Performance Metrics

You can track:
- **Task reward**: Accuracy on GSM8K problems
- **Sequence length**: Average response length
- **Advantages**: GRPO advantage statistics
- **Loss**: PPO loss values

All metrics are logged to WandB (if configured).

## Adjusting Training Time

If you want to adjust the training time:

### Make it Faster (~45 minutes)
- Reduce dataset: `MAX_TRAIN_SAMPLES = 300` (in script)
- Reduce epochs: `total_train_epochs: 1` (in YAML)
- Result: ~38 steps, ~45 minutes

### Make it Longer (~2-3 hours)
- Increase dataset: `MAX_TRAIN_SAMPLES = 1000` (in script)
- Increase epochs: `total_train_epochs: 3` (in YAML)
- Result: ~375 steps, ~3 hours

## Comparison with Other Configs

### vs Fast Training (20-30 min)
- ✅ **More data**: 500 vs 200 samples (2.5x more)
- ✅ **More epochs**: 2 vs 1 (better convergence)
- ✅ **Full quality**: n_samples=4, max_new_tokens=512
- ⚠️ **Longer time**: ~1-2 hours vs 20-30 minutes

### vs Full Training (5 days)
- ✅ **Much faster**: ~1-2 hours vs 5 days
- ⚠️ **Less data**: 500 vs 7,473 samples (15x less)
- ⚠️ **Fewer epochs**: 2 vs 5

## Expected Results

With 1-hour training, you should see:
- **Initial accuracy**: ~18-20% (base model)
- **After training**: ~25-30% (improved)
- **Improvement**: +5-10 percentage points

This is a good balance between training time and performance improvement!

## Tips

1. **Monitor WandB**: Watch the `task_reward` metric to track accuracy
2. **Check GPU usage**: Ensure GPU is being utilized efficiently
3. **Save checkpoints**: Checkpoints are saved after each epoch
4. **Evaluate**: Evaluation runs after each epoch automatically

## Troubleshooting

### Training takes longer than expected
- Check GPU utilization: `nvidia-smi`
- Monitor step time in logs
- Consider reducing `n_samples` to 2 if needed

### Out of memory
- Reduce `batch_size` to 4
- Reduce `max_new_tokens` to 256
- Enable `gradient_checkpointing: true`

### Want better results
- Increase `MAX_TRAIN_SAMPLES` to 1000
- Increase `total_train_epochs` to 3
- This will take ~2-3 hours but should improve accuracy

