# Quick Fix Commands

## The Issue

You hit MPS out of memory because the model is too large for batch_size=2 on 32GB RAM.

## The Fix: Run This Now

```bash
source venv/bin/activate

# Option 1: Use the provided script (easiest)
./examples/local_gsm8k/run_training.sh

# OR Option 2: Manual command
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.4
python examples/local_gsm8k/train_local_simple.py \
    --model ./models/DeepSeek-R1-Distill-Qwen-1.5B \
    --max-time 1800 \
    --batch-size 1 \
    --gradient-accumulation-steps 32 \
    --max-length 256 \
    --wandb
```

## What Changed?

1. **Batch size**: 2 → 1 (half the memory per step)
2. **Gradient accumulation**: 16 → 32 (keeps effective batch size at 32)
3. **Max length**: 512 → 256 (half the sequence length)
4. **Memory watermark**: Limited to 40% of available RAM

## Alternative: Use CPU (Slower but More Reliable)

If you still have issues, run on CPU:

```bash
python examples/local_gsm8k/train_local_simple.py \
    --model ./models/DeepSeek-R1-Distill-Qwen-1.5B \
    --device cpu \
    --max-time 1800 \
    --batch-size 2 \
    --gradient-accumulation-steps 16 \
    --wandb
```

## Settings Explained

- `--batch-size 1`: Process 1 sample at a time
- `--gradient-accumulation-steps 32`: Accumulate gradients for 32 steps before updating weights
- `--max-length 256`: Limit sequences to 256 tokens
- `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.4`: Use only 40% of available RAM

This should work! 🚀

