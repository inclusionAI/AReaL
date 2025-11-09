# Fast Training Guide (10-30 Minutes)

## Overview

The full training configuration (`gsm8k_grpo.yaml`) takes ~5 days to complete. This guide provides a **fast training setup** that completes in **20-30 minutes** while still showing meaningful improvement.

## Current vs Fast Configuration

| Setting | Full Training | Fast Training | Impact |
|---------|--------------|--------------|--------|
| **Epochs** | 5 | 1 | 5x faster |
| **Dataset Size** | 7,473 samples | 200 samples | 37x faster |
| **Steps per Epoch** | ~934 | ~25 | 37x faster |
| **Total Steps** | ~4,670 | ~25 | 187x faster |
| **n_samples** | 4 | 2 | 2x faster rollout |
| **max_new_tokens** | 512 | 256 | 2x faster generation |
| **Evaluation** | Every epoch | Disabled | Saves time |
| **Estimated Time** | ~5 days | ~20-30 min | 240x faster |

## Files Created

1. **`gsm8k_grpo_fast.yaml`** - Fast training configuration
2. **`gsm8k_grpo_fast.py`** - Fast training script (limits dataset to 200 samples)
3. **`TRAINING_SETTINGS_BREAKDOWN.md`** - Detailed analysis of current settings

## Running Fast Training

### Inside Docker Container

```bash
cd /workspace/AReaL

# Set WandB API key
export WANDB_API_KEY=$(cat examples/local_gsm8k/wandb/.wandb_api_key 2>/dev/null || echo "your-api-key")

# Run fast training
python3 -m areal.launcher.local examples/docker_gsm8k/gsm8k_grpo_fast.py \
    --config examples/docker_gsm8k/gsm8k_grpo_fast.yaml \
    experiment_name=gsm8k-grpo-fast \
    trial_name=trial0
```

### Expected Output

```
[FAST MODE] Limiting dataset from 7473 to 200 samples
[FAST MODE] Training on 200 samples
[FAST MODE] Steps per epoch: 25
[FAST MODE] Total steps: 25
[FAST MODE] Starting training: 25 total steps
[FAST MODE] Estimated time: ~25 minutes (at ~1 step/min)
```

## What You'll See

### Training Progress

- **25 steps total** (vs 4,670 in full training)
- **~1 step per minute** (faster due to optimizations)
- **~20-30 minutes total** training time

### Performance Metrics

Monitor in WandB or logs:
- `grpo_actor/task_reward/avg` - Accuracy (should improve from ~25% to ~30-35%)
- `grpo_actor/actor_loss/avg` - Training loss (should decrease)
- `grpo_actor/advantages/avg` - Policy advantages

### Expected Improvement

Even with just 200 samples and 1 epoch, you should see:
- **Accuracy improvement**: Base model ~25% → Trained model ~30-35%
- **Loss decrease**: Training loss should trend downward
- **Learning signal**: Model learns to generate better math solutions

## Key Optimizations in Fast Config

### 1. Reduced Dataset Size
```python
# In gsm8k_grpo_fast.py
MAX_TRAIN_SAMPLES = 200  # Instead of full 7,473
train_dataset = train_dataset.select(range(MAX_TRAIN_SAMPLES))
```

### 2. Reduced Epochs
```yaml
# In gsm8k_grpo_fast.yaml
total_train_epochs: 1  # Instead of 5
```

### 3. Faster Rollout
```yaml
gconfig:
  n_samples: 2  # Instead of 4 (half the samples per prompt)
  max_new_tokens: 256  # Instead of 512 (faster generation)
```

### 4. Disabled Evaluation
```yaml
evaluator:
  freq_epochs: null  # Disabled during fast training
```

## When to Use Fast vs Full Training

### Use Fast Training (`gsm8k_grpo_fast`) When:
- ✅ Testing code changes
- ✅ Debugging issues
- ✅ Quick iteration on hyperparameters
- ✅ Demonstrating the training pipeline
- ✅ Learning how GRPO works

### Use Full Training (`gsm8k_grpo`) When:
- ✅ Final model training
- ✅ Production deployment
- ✅ Research experiments
- ✅ Benchmarking performance

## Adjusting Fast Training Duration

You can easily adjust the speed by changing `MAX_TRAIN_SAMPLES` in `gsm8k_grpo_fast.py`:

```python
# Very fast (10 minutes): 100 samples
MAX_TRAIN_SAMPLES = 100  # ~12 steps

# Fast (20-30 minutes): 200 samples (default)
MAX_TRAIN_SAMPLES = 200  # ~25 steps

# Medium (1 hour): 400 samples
MAX_TRAIN_SAMPLES = 400  # ~50 steps

# Slower (2 hours): 800 samples
MAX_TRAIN_SAMPLES = 800  # ~100 steps
```

## Troubleshooting

### Training Still Too Slow?

1. **Check GPU utilization**: `nvidia-smi` (should be high)
2. **Reduce n_samples further**: Change `n_samples: 2` → `n_samples: 1` in YAML
3. **Reduce max_new_tokens**: Change `max_new_tokens: 256` → `max_new_tokens: 128`
4. **Increase batch_size**: If memory allows, `batch_size: 8` → `batch_size: 16`

### Not Seeing Improvement?

- **200 samples is minimal** - expect modest improvement (~5-10% accuracy gain)
- **For better results**: Use 400-800 samples (1-2 hours training)
- **Check logs**: Ensure `task_reward/avg` is increasing over time

## Next Steps After Fast Training

1. **Verify it works**: Check accuracy improved
2. **Test the model**: Use `test_trained_model.py`
3. **Scale up**: If results look good, try:
   - 400 samples (1 hour)
   - 800 samples (2 hours)
   - Full dataset (5 days)

## Summary

✅ **Fast training**: 20-30 minutes, 200 samples, 1 epoch  
✅ **Still meaningful**: Shows learning and improvement  
✅ **Great for**: Testing, debugging, quick iteration  
✅ **Files**: `gsm8k_grpo_fast.py` + `gsm8k_grpo_fast.yaml`

Run the fast training to see the pipeline in action, then scale up when ready!

