# 3-Hour Training Guide for GRPO

## Overview

This guide provides a **3-hour training configuration** that extends the 1-hour training with more data and epochs for better convergence and potentially higher accuracy.

## Configuration Summary

| Setting | 1-Hour | 3-Hour | Full (5 days) |
|---------|--------|--------|---------------|
| **Epochs** | 2 | 3 | 5 |
| **Dataset Size** | 500 samples | 1000 samples | 7,473 samples |
| **Steps per Epoch** | ~63 | ~125 | ~934 |
| **Total Steps** | ~126 | ~375 | ~4,670 |
| **n_samples** | 4 | 4 | 4 |
| **max_new_tokens** | 512 | 512 | 512 |
| **Batch Size** | 8 | 8 | 8 |
| **Estimated Time** | ~1-2 hours | ~3-4 hours | ~5 days |

## Key Parameters

### Dataset Size: 1000 Samples
- **Rationale**: 2x more data than 1-hour training for better coverage
- **Steps per epoch**: 1000 / 8 = ~125 steps
- **Total steps**: 3 epochs × 125 = ~375 steps

### Epochs: 3
- **Rationale**: Allows the model to see each sample three times
- Better convergence than 2 epochs, but faster than 5 epochs

### Generation Parameters
- **n_samples: 4**: Keeps full GRPO quality (4 solutions per problem)
- **max_new_tokens: 512**: Allows full reasoning chains

### Batch Size: 8
- Kept at 8 for stability and memory efficiency

## Files Created

1. **`gsm8k_grpo_3hour.yaml`** - 3-hour training configuration
2. **`gsm8k_grpo_3hour.py`** - Training script (limits dataset to 1000 samples)

## Running 3-Hour Training

### Inside Docker Container

```bash
cd /workspace/AReaL

# Set WandB API key
export WANDB_API_KEY=$(cat examples/local_gsm8k/wandb/.wandb_api_key 2>/dev/null || echo "your-api-key")

# Run 3-hour training
python3 -m areal.launcher.local examples/docker_gsm8k/gsm8k_grpo_3hour.py \
    --config examples/docker_gsm8k/gsm8k_grpo_3hour.yaml \
    experiment_name=gsm8k-grpo-3hour \
    trial_name=trial0
```

### Expected Output

```
[3-HOUR TRAINING MODE] Limiting dataset from 7473 to 1000 samples
[3-HOUR TRAINING MODE]
  Dataset size: 1000 samples
  Batch size: 8
  Steps per epoch: 125
  Total epochs: 3
  Total steps: 375
  Estimated time: ~375 minutes (~3.1 hours) at ~1 step/min
```

**Note**: The actual time may be closer to 3-4 hours depending on your hardware.

## What You'll See

### Training Progress

- **~375 steps total** (vs 126 in 1-hour, 4,670 in full)
- **~1 step per minute** (may vary based on hardware)
- **~3-4 hours total** training time
- **3 epochs** for better convergence

### Performance Metrics

You can track in WandB:
- **Task reward**: Accuracy on GSM8K problems
- **Sequence length**: Average response length
- **Advantages**: GRPO advantage statistics
- **Loss**: PPO loss values

All metrics are logged to WandB (if configured).

## Expected Results

With 3-hour training, you should see:
- **Initial accuracy**: ~18-20% (base model)
- **After training**: ~35-40% (improved from 1-hour's 32%)
- **Improvement**: +15-20 percentage points from base model

This should be better than the 1-hour training (32.05%) due to:
- ✅ More training data (1000 vs 500 samples)
- ✅ More epochs (3 vs 2)
- ✅ More training steps (375 vs 126)

## Comparison with Other Configs

### vs 1-Hour Training
- ✅ **More data**: 1000 vs 500 samples (2x more)
- ✅ **More epochs**: 3 vs 2 (better convergence)
- ✅ **More steps**: 375 vs 126 (3x more)
- ⚠️ **Longer time**: ~3-4 hours vs 1-2 hours

### vs Full Training (5 days)
- ✅ **Much faster**: ~3-4 hours vs 5 days
- ⚠️ **Less data**: 1000 vs 7,473 samples (7.5x less)
- ⚠️ **Fewer epochs**: 3 vs 5

## Tips

1. **Monitor WandB**: Watch the `task_reward` metric to track accuracy
2. **Check GPU usage**: Ensure GPU is being utilized efficiently
3. **Save checkpoints**: Checkpoints are saved after each epoch
4. **Evaluate**: Evaluation runs after each epoch automatically
5. **Check training curve**: Look for diminishing returns in WandB

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
- Increase `MAX_TRAIN_SAMPLES` to 2000
- Increase `total_train_epochs` to 5
- This will take ~6-8 hours but should improve accuracy

## After Training

### Evaluate the Model

```bash
# Test on full test set
python3 -m areal.launcher.local examples/docker_gsm8k/test_trained_model.py \
    --config examples/docker_gsm8k/gsm8k_grpo_3hour.yaml \
    experiment_name=gsm8k-grpo-3hour-eval \
    trial_name=full_test \
    rollout.experiment_name=gsm8k-grpo-3hour \
    rollout.trial_name=trial0
```

### Check Training Curves

1. Go to WandB: https://wandb.ai
2. Open project: `gsm8k-grpo-local`
3. Find your run: `gsm8k-grpo-3hour_trial0_train`
4. Check `grpo_actor/task_reward/avg` curve
5. Look for plateau to determine if diminishing returns reached

## Next Steps

After 3-hour training:
1. ✅ Evaluate on full test set
2. ✅ Check WandB training curves
3. ✅ Compare with 1-hour training results
4. ✅ Decide if more training is needed

Good luck with your 3-hour training run! 🚀

