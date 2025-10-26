# 2-Hour Training Run - Active 🚀

## Optimized Configuration

**Settings for 2-hour budget:**

- **Model**: Qwen/Qwen2.5-0.5B-Instruct
- **Dataset**: 1,500 GSM8K samples (3x more than short run)
- **Device**: CPU (MPS disabled to avoid memory errors)
- **Batch Size**: 2 (increased from 1)
- **Gradient Accumulation**: 16 steps
- **Epochs**: 3
- **Learning Rate**: 3e-5
- **Max Length**: 128 tokens
- **Time Budget**: 2 hours

## Progress

- **Steps per epoch**: ~141
- **Total steps**: ~423 (3 epochs)
- **Time per step**: ~6 seconds (from CPU training)
- **Estimated total**: ~45 minutes ✅

## Monitor

```bash
# Watch live training
tail -f /tmp/training_2hour.log

# Check process
ps aux | grep train_hf_trainer | grep 6351

# View W&B
# https://wandb.ai/tong-zhao-georgia-institute-of-technology/huggingface/runs/s8nhvvnx
```

## Expected Results

Compared to 500-sample run:
- 3x more training data
- Should show better loss convergence
- Better coverage of problem types
- Model saved to: `outputs/gsm8k-2hour/`

## What's Different

From short run (500 samples):
- ✅ More data (1,500 vs 500)
- ✅ Larger batch size (2 vs 1) - faster
- ✅ Same effective batch size (16 grad acc)
- ✅ Shorter sequences (128 vs 256) - faster
- ✅ CPU training (stable, no MPS errors)

---

**Started**: 2024-10-26 14:23  
**PID**: 6351  
**Expected completion**: ~15:10 (45 min)

