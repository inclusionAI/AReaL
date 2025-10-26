# Recommended Training Command

## The Problem

MPS (Metal GPU) on Mac M2 is hitting memory limits even with optimizations.

## The Solution: Use CPU

CPU training is slower but **more reliable** and will complete successfully.

### Quick Start:

```bash
source venv/bin/activate

# Run this command (slower but will work!)
python examples/local_gsm8k/train_local_simple.py \
    --model ./models/DeepSeek-R1-Distill-Qwen-1.5B \
    --max-time 1800 \
    --batch-size 1 \
    --gradient-accumulation-steps 32 \
    --max-length 256 \
    --device cpu \
    --wandb
```

### OR use the script:

```bash
source venv/bin/activate
./examples/local_gsm8k/train_cpu.sh
```

## What This Will Do

✅ Train for up to 30 minutes  
✅ Use CPU (stable, no memory issues)  
✅ Save checkpoints to `outputs/`  
✅ Track with W&B  
✅ Process 500 GSM8K samples  

## Expected Time

- **CPU**: ~30-45 minutes for full run
- Will be slower than MPS but won't crash

## Check Progress

You can monitor the training:

```bash
# Watch the log
tail -f /tmp/training_run.log

# Or check W&B dashboard
# https://wandb.ai/tong-zhao-georgia-institute-of-technology/areal-gsm8k-mac
```

## When Training Completes

Test your model:

```bash
python examples/local_gsm8k/test_model.py \
    --model ./outputs/gsm8k-local \
    --max-samples 10
```

## Alternative: Much Smaller Model

If you want faster training and less memory issues:

```bash
# Train on smaller model
python examples/local_gsm8k/train_local_simple.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --max-time 1800 \
    --batch-size 4 \
    --gradient-accumulation-steps 8 \
    --device mps
```

This will train much faster!

