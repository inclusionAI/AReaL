# How to Evaluate Your Fast-Trained Model

## Quick Evaluation (50 Samples - Recommended)

For fast evaluation on just 50 test samples (~2-5 minutes):

```bash
# Inside Docker container
cd /workspace/AReaL

# Set WandB API key (optional, for logging)
export WANDB_API_KEY=$(cat examples/local_gsm8k/wandb/.wandb_api_key 2>/dev/null || echo "your-api-key")

# Run fast evaluation (50 samples only)
python3 -m areal.launcher.local examples/docker_gsm8k/test_trained_model_fast.py \
    --config examples/docker_gsm8k/gsm8k_grpo_fast.yaml \
    experiment_name=gsm8k-grpo-fast \
    trial_name=trial0 \
    rollout.experiment_name=gsm8k-grpo-fast \
    rollout.trial_name=trial0
```

**Time**: ~2-5 minutes  
**Samples**: 50 test samples  
**Use case**: Quick iteration and testing

## Full Evaluation (All 1,319 Samples)

For complete evaluation on the full test set (~10-20 minutes):

```bash
# Inside Docker container
cd /workspace/AReaL

python3 -m areal.launcher.local examples/docker_gsm8k/test_trained_model.py \
    --config examples/docker_gsm8k/gsm8k_grpo_fast.yaml \
    experiment_name=gsm8k-grpo-fast \
    trial_name=trial0 \
    rollout.experiment_name=gsm8k-grpo-fast \
    rollout.trial_name=trial0
```

**Time**: ~10-20 minutes  
**Samples**: 1,319 test samples (full test set)  
**Use case**: Final evaluation before deployment

## Alternative: Limit Full Test Script

You can also limit the full test script using an environment variable:

```bash
# Evaluate on 50 samples using the full test script
export MAX_TEST_SAMPLES=50
python3 -m areal.launcher.local examples/docker_gsm8k/test_trained_model.py \
    --config examples/docker_gsm8k/gsm8k_grpo_fast.yaml \
    experiment_name=gsm8k-grpo-fast \
    trial_name=trial0 \
    rollout.experiment_name=gsm8k-grpo-fast \
    rollout.trial_name=trial0
```

## What to Expect

### Fast Evaluation Output (50 samples)

```
[FAST EVAL] Limiting test set from 1319 to 50 samples
[FAST EVAL] Evaluating on 50 test samples (subset)
[FAST EVAL] Using model from: gsm8k-grpo-fast/trial0
[FAST EVAL] Estimated time: ~2-5 minutes
[FAST EVAL] Submitted 50 evaluation tasks. Waiting for completion...

================================================================================
FAST EVALUATION RESULTS (50 samples)
================================================================================
[Detailed stats table]

================================================================================
ACCURACY: 32.00% (on 50 test samples)
================================================================================

Note: This is accuracy on a subset. Full test set has 1,319 samples.
For full evaluation, use: test_trained_model.py
```

## Expected Results

For fast training (200 samples, 1 epoch):
- **Base model**: ~25-30% accuracy
- **Trained model**: ~30-35% accuracy
- **Improvement**: ~5-10% accuracy gain

**Note**: Accuracy on 50 samples may vary ±5% compared to full test set due to small sample size.

## Where Your Model is Saved

Your trained model checkpoints are saved at:

```
./outputs/grpo/checkpoints/root/gsm8k-grpo-fast/trial0/default/
  ├── epoch0epochstep0globalstep1/
  ├── epoch0epochstep1globalstep2/
  ├── ...
  └── epoch0epochstep24globalstep25/  (last checkpoint)
```

The evaluation script automatically loads the latest checkpoint from this location.

## Quick Check: View Training Accuracy

You can also check the accuracy from your training logs:

```bash
# Inside Docker container
grep "task_reward/avg" /workspace/AReaL/outputs/grpo/logs/root/gsm8k-grpo-fast/trial0/trainer.log | tail -5
```

Or check WandB:
- Visit: https://wandb.ai/tong-zhao-georgia-institute-of-technology/gsm8k-grpo-local
- Find run: `gsm8k-grpo-fast_trial0_train`
- Look at `grpo_actor/task_reward/avg` metric

## Metrics to Check

1. **`eval-rollout/task_reward/avg`**: This is your **accuracy** (0.0-1.0, multiply by 100 for %)
2. **`eval-rollout/seq_len/avg`**: Average sequence length of generated answers
3. **`eval-rollout/final_reward/avg`**: Final reward (should match task_reward for GSM8K)

## Troubleshooting

### "Model not found" error

Check if checkpoints exist:
```bash
# Inside Docker container
ls -la /workspace/AReaL/outputs/grpo/checkpoints/root/gsm8k-grpo-fast/trial0/default/
```

If empty, the training might not have saved checkpoints. Check training logs.

### Want different number of test samples?

Edit `test_trained_model_fast.py` and change:
```python
MAX_TEST_SAMPLES = 50  # Change to your desired number
```

## Summary

✅ **Fast evaluation (50 samples)**: Use `test_trained_model_fast.py` - ~2-5 minutes  
✅ **Full evaluation (1,319 samples)**: Use `test_trained_model.py` - ~10-20 minutes  
✅ **Model location**: `outputs/grpo/checkpoints/root/gsm8k-grpo-fast/trial0/default/`  
✅ **Expected accuracy**: ~30-35% (modest improvement from base ~25-30%)

Run the fast evaluation command above to quickly see your model's performance!
