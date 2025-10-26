# Local GSM8K Fine-tuning on Mac M2 - Start Here

## What's Been Set Up

This directory contains a **simplified, standalone training setup** for fine-tuning language models on the GSM8K dataset. Unlike the main AReaL framework which requires distributed GPU clusters, this script works locally on your Mac M2.

### Files Created

- **`train_local_simple.py`** - Main training script
- **`test_model.py`** - Model testing and comparison script
- **`README.md`** - Detailed documentation
- **`QUICKSTART.md`** - Quick start guide
- **`requirements.txt`** - Dependencies

### What You Can Do

✅ Fine-tune DeepSeek-R1-Distill-Qwen-1.5B (or any model)  
✅ Train on GSM8K dataset  
✅ Complete training in under 30 minutes for testing  
✅ Compare model performance before/after training  
✅ Track progress with W&B (Weights & Biases)  
✅ Run on Mac M2 with MPS acceleration  

## Quick Start (Copy & Paste)

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Start a 30-minute test run
python examples/local_gsm8k/train_local_simple.py \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --output-dir ./outputs/gsm8k-test \
    --max-time 1800 \
    --batch-size 2 \
    --gradient-accumulation-steps 16 \
    --wandb

# 3. Test the model before training
python examples/local_gsm8k/test_model.py \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --max-samples 5

# 4. Test after training
python examples/local_gsm8k/test_model.py \
    --model ./outputs/gsm8k-test \
    --max-samples 5
```

## Hardware Requirements

- **Mac M2** (32GB RAM as you have)
- **Python 3.10.11** (already set up)
- **Virtual environment** (venv already created)

## Expected Training Time

- **30-minute test**: ~500 samples, ~188 steps
- **Full training**: Can run longer by adjusting `--max-time`

## Memory Usage

- Batch size 2 + gradient accumulation 16 = effective batch size 32
- Should fit comfortably in 32GB RAM
- Uses gradient checkpointing to save memory

## What Gets Saved

- Model checkpoints: `./outputs/gsm8k-test/`
- W&B logs: Automatically synced (if logged in)
- Generated text samples
- Training metrics

## Model Comparison

After training, you can compare:

```bash
python examples/local_gsm8k/test_model.py \
    --compare \
    --base-model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --trained-model ./outputs/gsm8k-test
```

This will:
1. Test base model on GSM8K
2. Test trained model on GSM8K  
3. Show accuracy comparison
4. Save results to `model_comparison.json`

## Next Steps

1. **Run the training** (see QUICKSTART.md)
2. **Test and compare** models
3. **Extend training** if results look good
4. **Submit to HuggingFace leaderboard** for GSM8K

## Troubleshooting

### If you get import errors:
The script has fallback implementations, so you don't need the full AReaL installation.

### If you run out of memory:
```bash
--batch-size 1 --gradient-accumulation-steps 32
```

### If training is slow:
- This is expected on CPU/MPS, but faster than CPU alone
- First run downloads the model (~3GB)

## W&B Integration

To use Weights & Biases:

```bash
wandb login
```

Then training will automatically log:
- Training loss
- Learning rate
- Training metrics
- Model checkpoints

## Questions?

See `README.md` for detailed documentation or `QUICKSTART.md` for step-by-step instructions.

