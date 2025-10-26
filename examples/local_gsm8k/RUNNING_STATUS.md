# 🚀 Training Currently Running!

## Status: Active

Full training is currently running in the background on your machine.

### Training Details

- **Model**: Qwen/Qwen2.5-0.5B-Instruct  
- **Dataset**: 500 GSM8K samples
- **Device**: MPS (Apple Silicon)
- **Epochs**: 3
- **Batch Size**: 2
- **Gradient Accumulation**: 16 steps (effective batch size = 32)
- **Max Sequence Length**: 256 tokens
- **Time Limit**: 30 minutes (1800 seconds)

### Progress

- **Total Steps per Epoch**: ~48 steps
- **Estimated Time**: ~20-25 minutes
- **Training Speed**: ~8 seconds per step

### W&B Tracking

Monitor progress here: https://wandb.ai/tong-zhao-georgia-institute-of-technology/huggingface/runs/dpftd5fa

### Check Status

```bash
# View running logs
tail -f /tmp/training_full.log

# Check if still running
ps aux | grep train_hf_trainer

# View W&B output
ls -lrt wandb/ | tail -n 5
```

### What Happens When Complete

1. Model will be saved to: `outputs/gsm8k-hf-trainer/`
2. Final checkpoint will be created
3. You can test it with:
   ```bash
   python examples/local_gsm8k/test_model.py --model ./outputs/gsm8k-hf-trainer
   ```

### Key Improvements in This Training

✅ Using HuggingFace Trainer (proper loss masking)  
✅ Proper label masking with -100 for question tokens  
✅ Data collator handles batching/padding correctly  
✅ W&B tracking enabled  
✅ Memory-efficient with gradient checkpointing  

### Next Steps After Training

1. Test the model on GSM8K samples
2. Compare base vs trained model accuracy
3. Optionally train longer or on more data

---

**Started**: $(date)  
**Expected Completion**: ~25 minutes from start

