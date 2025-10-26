# 2-Hour Training Run 🚀

## Configuration

- **Model**: Qwen/Qwen2.5-0.5B-Instruct
- **Dataset**: 2,000 GSM8K samples (out of 7,473 total)
- **Device**: CPU (stable, no memory issues)
- **Batch Size**: 1
- **Gradient Accumulation**: 32 steps
- **Epochs**: 5
- **Learning Rate**: 3e-5
- **Max Length**: 256 tokens
- **Time Budget**: 2 hours (7,200 seconds)

## Progress

- **Steps per epoch**: ~315
- **Total steps**: ~1,575 steps
- **Time per step**: ~4.5 seconds (estimated)
- **Total time**: ~2 hours

## Monitor Progress

```bash
# Watch live logs
tail -f /tmp/training_2hour.log

# Check if running
ps aux | grep train_hf_trainer

# View W&B dashboard
# https://wandb.ai/tong-zhao-georgia-institute-of-technology/huggingface/runs/eg7murq6
```

## Expected Improvements

Compared to previous 500-sample training:
- ✅ **4x more data** (500 → 2,000 samples)
- ✅ **More epochs** (3 → 5 epochs)
- ✅ **Better coverage** of GSM8K problem types
- ✅ **Lower learning rate** (3e-5 vs 5e-5) for more stable training

## What to Expect

- Start: Loss ~0.9-1.0
- Middle: Loss ~0.4-0.5
- End: Loss ~0.2-0.3 (lower = better)

## Output Location

Model will be saved to: `outputs/gsm8k-2hour/`

## After Training

Test with:
```bash
python examples/local_gsm8k/test_model.py --model ./outputs/gsm8k-2hour --max-samples 20
```

---

**Started**: 2024-10-26 14:01  
**PID**: 5487  
**Expected completion**: ~16:00 (2 hours)

