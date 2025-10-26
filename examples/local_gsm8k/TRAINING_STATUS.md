# Training Status

## Current Run

✅ **Training is RUNNING on CPU**

- **Started**: Just now  
- **Device**: CPU (stable, no memory issues)  
- **Duration**: Up to 30 minutes
- **Model**: DeepSeek-R1-Distill-Qwen-1.5B
- **Dataset**: First 500 samples from GSM8K train set
- **W&B**: https://wandb.ai/tong-zhao-georgia-institute-of-technology/areal-gsm8k-mac

## Monitor Progress

### Option 1: Watch the log
```bash
tail -f /tmp/training_cpu.log
```

### Option 2: Check W&B Dashboard
```bash
# Open in browser:
# https://wandb.ai/tong-zhao-georgia-institute-of-technology/areal-gsm8k-mac
```

### Option 3: Check if still running
```bash
ps aux | grep train_local_simple
```

## When Training Completes

1. **Check output**:
   ```bash
   ls -lh outputs/gsm8k-local/
   ```

2. **Test the model**:
   ```bash
   python examples/local_gsm8k/test_model.py \
       --model ./outputs/gsm8k-local \
       --max-samples 10
   ```

3. **Compare with base model**:
   ```bash
   python examples/local_gsm8k/test_model.py \
       --compare \
       --base-model ./models/DeepSeek-R1-Distill-Qwen-1.5B \
       --trained-model ./outputs/gsm8k-local
   ```

## Expected Timeline

- **CPU training**: ~30-45 minutes total
- **Memory**: Stable on CPU (no crashes)
- **Output**: Checkpoints saved every 50 steps

## Check Training Status Right Now

```bash
tail -f /tmp/training_cpu.log
```

