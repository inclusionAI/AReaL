# Local GSM8K Training Setup

Complete training setup for finetuning LLMs on GSM8K dataset locally on Mac M2.

## Quick Start

```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Run training (2-hour budget)
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

# 3. Test trained model
python examples/local_gsm8k/test_model.py --model ./outputs/gsm8k-2hour
```

## Files

- **`train_hf_trainer.py`** - Main training script (HuggingFace Trainer)
- **`test_model.py`** - Model testing and evaluation
- **`START_HERE.md`** - Quick start guide
- **`QUICKSTART.md`** - Step-by-step instructions
- **`NAN_FIX_EXPLAINED.md`** - Technical details on NaN loss fix
- **`HF_TRAINER_SUCCESS.md`** - Why HuggingFace Trainer works

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

```bash
tail -f /tmp/training_2hour.log
```

## Test Results

After training completes, run:
```bash
python examples/local_gsm8k/test_model.py --model ./outputs/gsm8k-2hour --max-samples 20
```
