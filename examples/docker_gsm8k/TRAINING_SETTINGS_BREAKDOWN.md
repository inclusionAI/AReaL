# Training Settings Breakdown

## Current Configuration Analysis

### Current Settings (gsm8k_grpo.yaml)

**Training Duration:**
- `total_train_epochs: 5`
- Dataset: `openai/gsm8k` (full training set ~7,473 samples)
- `batch_size: 8`
- Steps per epoch: ~7,473 / 8 = **~934 steps**
- Total steps: 5 epochs × 934 = **~4,670 steps**

**Current Speed:**
- 15 steps in 10 minutes = **0.67 steps/minute**
- At this rate, full training would take: **~7,000 minutes = ~116 hours = ~5 days!**

### What's Taking So Long?

1. **Large Dataset**: Full GSM8K training set (7,473 samples)
2. **Many Epochs**: 5 epochs
3. **Slow Rollout**: Each step requires:
   - Generating 4 samples per prompt (`n_samples: 4`)
   - Up to 512 tokens per sample (`max_new_tokens: 512`)
   - Weight update via disk (15-20 seconds per step)
4. **Evaluation**: Runs after each epoch

### Time Breakdown Per Step (~40 seconds/step)

- Rollout (generation): ~15-20 seconds
- Weight update (disk I/O): ~15-20 seconds  
- PPO update: ~5-10 seconds
- Logging/stats: ~1-2 seconds

## Fast Training Configuration

For 10-30 minute training, we need:

**Target: 20-30 steps total**
- At 0.67 steps/min: 30 steps = ~45 minutes
- But we can optimize to get ~1-1.5 steps/min = 20-30 minutes

**Optimizations:**
1. **Reduce dataset**: Use ~200 samples (25 steps/epoch)
2. **Reduce epochs**: 1 epoch
3. **Reduce samples per prompt**: 4 → 2 (`n_samples: 2`)
4. **Reduce max tokens**: 512 → 256 (`max_new_tokens: 256`)
5. **Skip evaluation**: Disable during fast training

**Expected:**
- 1 epoch × 25 steps = **25 steps total**
- At optimized speed (~1 step/min): **~25 minutes**
- Still enough to see meaningful improvement!

