# How to Test Your Trained Model

## Model Checkpoints Location

Models are automatically saved during training to:
```
./outputs/grpo/checkpoints/root/{experiment_name}/{trial_name}/default/epoch{epoch}epochstep{step}globalstep{global_step}/
```

For your current training run:
```
./outputs/grpo/checkpoints/root/gsm8k-grpo-docker/trial0/default/epoch0epochstep0globalstep1/
```

Each checkpoint contains:
- Model weights (`model.safetensors` or `pytorch_model.bin`)
- Tokenizer files
- Config files

## Accuracy Tracking During Training

**Good news**: Accuracy is **already being tracked** during training!

### Where to Find It

1. **In Training Logs**:
   - Look for `grpo_actor/task_reward/avg` in the stats table
   - This is the accuracy (0.0 = 0%, 1.0 = 100%)
   - Currently showing: **25-28%** (0.25-0.28)

2. **In WandB Dashboard**:
   - Visit: https://wandb.ai/tong-zhao-georgia-institute-of-technology/gsm8k-grpo-local
   - Look for run: `gsm8k-grpo-docker_trial0_train`
   - Metric: `grpo_actor/task_reward/avg` (this is accuracy!)

3. **Evaluation During Training**:
   - The training script runs evaluation after each epoch
   - Check stats table for `eval-rollout/task_reward/avg`

### Current Training Stats

From the latest logs, you can see:
- `grpo_actor/task_reward/avg`: ~0.25-0.28 (25-28% accuracy)
- This means about 25-28% of generated answers are correct

## Testing After Training Completes

### Option 1: Use AReaL's Built-in Evaluation Script

```bash
# Inside Docker container
cd /workspace/AReaL

python3 -m areal.launcher.local examples/math/gsm8k_eval.py \
    --config examples/docker_gsm8k/gsm8k_grpo.yaml \
    experiment_name=gsm8k-grpo-docker \
    trial_name=trial0 \
    rollout.experiment_name=gsm8k-grpo-docker \
    rollout.trial_name=trial0
```

This will:
- Load the model from the latest checkpoint
- Evaluate on the full GSM8K test set
- Print accuracy statistics

### Option 2: Use Custom Test Script

I've created `examples/docker_gsm8k/test_trained_model.py` for testing:

```bash
# Inside Docker container
cd /workspace/AReaL

python3 -m areal.launcher.local examples/docker_gsm8k/test_trained_model.py \
    --config examples/docker_gsm8k/gsm8k_grpo.yaml \
    experiment_name=gsm8k-grpo-docker \
    trial_name=trial0 \
    rollout.experiment_name=gsm8k-grpo-docker \
    rollout.trial_name=trial0
```

### Option 3: Test Specific Checkpoint

To test a specific checkpoint (e.g., after epoch 3):

```bash
# Find checkpoint paths
docker exec areal-grpo ls -la /workspace/AReaL/outputs/grpo/checkpoints/root/gsm8k-grpo-docker/trial0/default/

# Test specific checkpoint by modifying config or using checkpoint path
# (You may need to copy checkpoint to a model directory first)
```

## Understanding the Metrics

### During Training

- **`grpo_actor/task_reward/avg`**: Training accuracy (on training samples)
- **`eval-rollout/task_reward/avg`**: Validation accuracy (on test samples)

### Reward Function

The `gsm8k_reward_fn` returns:
- `1` if the generated answer is correct (matches ground truth)
- `0` if incorrect

So `task_reward/avg` of `0.25` = **25% accuracy**

## Quick Accuracy Check

You can check current accuracy from logs:

```bash
# Inside Docker container
grep "task_reward/avg" /workspace/AReaL/outputs/grpo/logs/root/gsm8k-grpo-docker/trial0/trainer.log | tail -5
```

Or from WandB:
- Go to: https://wandb.ai/tong-zhao-georgia-institute-of-technology/gsm8k-grpo-local
- Find your run
- Look at `grpo_actor/task_reward/avg` metric over time

## Model Files Location

Models are saved in:
```
outputs/grpo/checkpoints/root/gsm8k-grpo-docker/trial0/default/
  ├── epoch0epochstep0globalstep1/
  │   ├── config.json
  │   ├── tokenizer files
  │   └── model files
  ├── epoch0epochstep1globalstep2/
  └── ...
```

To use a checkpoint as a model path:
```python
checkpoint_path = "./outputs/grpo/checkpoints/root/gsm8k-grpo-docker/trial0/default/epoch0epochstep0globalstep1"
model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
```

## Summary

✅ **Accuracy IS being tracked**: Check `grpo_actor/task_reward/avg` in logs/WandB  
✅ **Models ARE being saved**: Checkpoints saved after each step (freq_epochs=1)  
✅ **Evaluation runs during training**: After each epoch on validation set  
✅ **Test scripts available**: Use `gsm8k_eval.py` or `test_trained_model.py` after training

