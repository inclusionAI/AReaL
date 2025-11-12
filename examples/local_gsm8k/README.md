# Local GSM8K Training Setup

Complete training setup for finetuning LLMs on GSM8K dataset locally on Mac M2.

## Quick Start

### Option 1: Using the training script (Recommended)

```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Run training with defaults (2-hour budget, 1500 samples)
bash examples/local_gsm8k/train.sh

# Or customize via environment variables:
MAX_SAMPLES=1000 BATCH_SIZE=4 bash examples/local_gsm8k/train.sh

# Or pass arguments directly:
bash examples/local_gsm8k/train.sh --max-samples 1000 --batch-size 4 --no-wandb
```

### Option 2: Using Python directly

```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Run training
python examples/local_gsm8k/train_hf_trainer.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --max-samples 1500 \
    --batch-size 2 \
    --gradient-accumulation-steps 16 \
    --num-epochs 3 \
    --learning-rate 3e-5 \
    --max-length 128 \
    --max-time 7200 \
    --output-dir ./outputs/gsm8k-2hour
```

## Files

- **`train.sh`** - Simple bash script to run training (recommended)
- **`train_hf_trainer.py`** - Main training script (HuggingFace Trainer)
- **`test_model.py`** - Test and compare model performance (base vs trained)
- **`HF_TRAINER_SUCCESS.md`** - Why HuggingFace Trainer works
- **`load_wandb_key.py`** - Utility to load W&B API key from local file
- **`requirements.txt`** - Python dependencies

## What Was Fixed

1. ✅ **NaN Loss Issue**: Fixed by switching from manual loss computation to HuggingFace Trainer
2. ✅ **MPS Memory Errors**: Resolved by forcing CPU training
3. ✅ **Loss Masking**: Proper -100 labels for question tokens
4. ✅ **Training Stability**: No more exploding gradients or NaN

## Training Results

### Short Run (500 samples, 3 epochs, ~5 min)
- Loss: 0.485
- Generates: Step-by-step reasoning
- Accuracy: 0% (answer format mismatch)

### Long Run (1,500 samples, 3 epochs, ~45 min)
- Currently running...
- Better data coverage
- More stable training

## Key Learnings

1. **HuggingFace Trainer > Manual Implementation**: Battle-tested, handles edge cases
2. **CPU > MPS for local training**: MPS has memory limits on 32GB RAM
3. **Small sequences faster**: 128 vs 256 tokens saves memory and time
4. **Loss masking critical**: Must ignore question tokens in loss

## Monitor Training

Training progress is logged to console and optionally to W&B if enabled.

## Output

Trained models are saved to the specified `--output-dir`:
- Model weights and tokenizer
- Training checkpoints (if `--save-steps` is set)
- Training logs

## Testing Trained Models

After training, test your model:

```bash
# Test a single model (quick test with 20 samples)
python examples/local_gsm8k/test_model.py --model ./outputs/gsm8k-training --max-samples 20

# Test on FULL GSM8K test set (all 1319 samples)
python examples/local_gsm8k/test_model.py --model ./outputs/gsm8k-training --all

# Or use -1 to test all samples
python examples/local_gsm8k/test_model.py --model ./outputs/gsm8k-training --max-samples -1

# Compare base model vs trained model (full test set)
python examples/local_gsm8k/test_model.py \
    --compare \
    --base-model Qwen/Qwen2.5-0.5B-Instruct \
    --trained-model ./outputs/gsm8k-training \
    --all
```

The test script will:
- Evaluate on GSM8K test set (full 1319 samples with `--all` or `--max-samples -1`)
- Extract and compare numerical answers using multiple methods
- Save detailed logs to `examples/local_gsm8k/logs/`
- Save comparison results to `model_comparison.json` (when using `--compare`)
