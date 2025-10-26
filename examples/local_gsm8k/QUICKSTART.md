# Quick Start Guide

## Setup

1. **Activate your virtual environment:**
   ```bash
   source venv/bin/activate
   ```

2. **Verify dependencies are installed:**
   ```bash
   pip list | grep -E "torch|transformers|datasets|wandb"
   ```

## Running Training (30-minute test)

```bash
python examples/local_gsm8k/train_local_simple.py \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --output-dir ./outputs/gsm8k-test \
    --max-time 1800 \
    --batch-size 2 \
    --gradient-accumulation-steps 16
```

This will:
- Train on GSM8K dataset (first 500 samples)
- Run for maximum 30 minutes
- Save checkpoints to `./outputs/gsm8k-test`
- Track progress with W&B (if available)

## Test the Model

### Before Training
```bash
python examples/local_gsm8k/test_model.py \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --max-samples 5
```

### After Training
```bash
python examples/local_gsm8k/test_model.py \
    --model ./outputs/gsm8k-test \
    --max-samples 5
```

### Compare Both
```bash
python examples/local_gsm8k/test_model.py \
    --compare \
    --base-model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --trained-model ./outputs/gsm8k-test \
    --max-samples 10
```

## Extended Training (Full Dataset)

For a more complete training run:

```bash
python examples/local_gsm8k/train_local_simple.py \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --output-dir ./outputs/gsm8k-full \
    --max-epochs 5 \
    --max-time 14400 \
    --batch-size 4 \
    --gradient-accumulation-steps 8
```

This will:
- Train for up to 4 hours
- Use the full training set (or as much as memory allows)
- Save intermediate checkpoints

## Memory Tips

### If you run out of memory:
- Reduce batch size: `--batch-size 1`
- Increase gradient accumulation: `--gradient-accumulation-steps 32`
- Reduce max length: `--max-length 256`

### To speed up training:
- Use smaller model first for testing
- Reduce dataset size in the script (edit the `select(range(500, ...))` line)

## Troubleshooting

### Import Error for areal
This is expected! The script has fallback implementations. Just proceed.

### CUDA/GPU errors
For M2 Mac, the script automatically uses MPS. If you see errors, try:
```bash
--device cpu
```

### WandB Login
If you want to use W&B:
```bash
wandb login
```

### Model Download
On first run, the model will be downloaded from HuggingFace. This can take a few minutes.

## Next Steps

1. **Evaluate on test set**: After training completes, test on the full GSM8K test set
2. **Submit to leaderboard**: Upload your model to HuggingFace
3. **Fine-tune hyperparameters**: Try different learning rates, batch sizes, etc.

## Example Output

```
Device: mps
Loading model from deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
Enabled gradient checkpointing
Loading GSM8K dataset...
Training on 500 samples
Starting training for 3 epochs
Total steps: 188
Steps per epoch: 500
Epoch 1/3: 100%|████████████████████| 500/500 [10:23<00:00, loss=2.345]
Epoch 1 average loss: 2.345
Saved checkpoint to ./outputs/gsm8k-test/checkpoint-step_50
...
Training completed!
```

