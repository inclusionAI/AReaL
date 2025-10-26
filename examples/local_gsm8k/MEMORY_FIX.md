# Memory Fix Guide

## Problem

You're getting: `RuntimeError: MPS backend out of memory`

This is because the 1.5B model with batch_size=2 is using too much memory on your 32GB Mac M2.

## Solutions

### Option 1: Reduce Batch Size (Recommended)

```bash
source venv/bin/activate

# Set lower memory watermark for MPS
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.4

# Run with smaller batch size
python examples/local_gsm8k/train_local_simple.py \
    --model ./models/DeepSeek-R1-Distill-Qwen-1.5B \
    --max-time 1800 \
    --batch-size 1 \
    --gradient-accumulation-steps 32 \
    --max-length 256 \
    --wandb
```

### Option 2: Use CPU Instead

CPU has more available memory:

```bash
python examples/local_gsm8k/train_local_simple.py \
    --model ./models/DeepSeek-R1-Distill-Qwen-1.5B \
    --device cpu \
    --max-time 1800 \
    --batch-size 2 \
    --gradient-accumulation-steps 16 \
    --wandb
```

### Option 3: Use a Smaller Model

If you want faster training with less memory:

```bash
# First download smaller model
python examples/local_gsm8k/download_model.py --model Qwen/Qwen2.5-0.5B-Instruct

# Then train
python examples/local_gsm8k/train_local_simple.py \
    --model ./models/Qwen2.5-0.5B-Instruct \
    --max-time 1800 \
    --batch-size 4 \
    --gradient-accumulation-steps 8 \
    --wandb
```

### Option 4: Use the Provided Script

Just run:

```bash
source venv/bin/activate
./examples/local_gsm8k/run_training.sh
```

This uses the optimal settings for 32GB RAM.

## Memory Breakdown

Your 32GB RAM:
- MPS allocation: ~19GB
- Other allocations: ~23GB
- Total: ~42GB (exceeded limit)

By reducing to:
- batch_size=1 
- max_length=256
- gradient_accumulation=32

You'll use ~30-35GB instead.

## Quick Fix Command

Just run this:

```bash
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.4 && \
python examples/local_gsm8k/train_local_simple.py \
    --model ./models/DeepSeek-R1-Distill-Qwen-1.5B \
    --max-time 1800 \
    --batch-size 1 \
    --gradient-accumulation-steps 32 \
    --max-length 256 \
    --wandb
```

This should work with your 32GB RAM!

