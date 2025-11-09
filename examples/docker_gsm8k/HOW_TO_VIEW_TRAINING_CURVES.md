# How to View Training Curves and Check for Diminishing Returns

## Overview

Training metrics are logged to **Weights & Biases (WandB)** during training. You can view the training curves online to analyze if you've reached diminishing returns.

## Method 1: WandB Online Dashboard (Recommended)

### Access Your WandB Dashboard

1. **Go to WandB website**: https://wandb.ai
2. **Login** with your WandB account (the one associated with API key: `e1adc5be02c03fd34828c84b1ece937e0c2feb6e`)
3. **Navigate to your project**: `gsm8k-grpo-local`
4. **Find your run**: Look for run name containing `gsm8k-grpo-1hour_trial1109_1`

### Key Metrics to Monitor

In WandB, look for these metrics to assess training progress:

#### 1. **Task Reward (Accuracy)**
- **Metric name**: `rollout/task_reward/avg` or `grpo_actor/task_reward/avg`
- **What it shows**: Model accuracy on training problems
- **What to look for**:
  - ✅ **Good**: Steady increase over time
  - ⚠️ **Plateau**: Flat line = diminishing returns reached
  - ❌ **Bad**: Decreasing = overfitting or instability

#### 2. **PPO Loss**
- **Metric name**: `grpo_actor/ppo_loss/avg`
- **What it shows**: Policy optimization loss
- **What to look for**:
  - ✅ **Good**: Decreasing or stable
  - ⚠️ **Warning**: Increasing = training instability

#### 3. **Advantages**
- **Metric name**: `grpo_actor/advantages/avg`
- **What it shows**: GRPO advantage values
- **What to look for**:
  - ✅ **Good**: Positive values (model learning)
  - ⚠️ **Warning**: Near zero = no learning signal

#### 4. **KL Divergence** (if using reference model)
- **Metric name**: `grpo_actor/kl_rewards/avg`
- **What it shows**: Distance from reference model
- **What to look for**:
  - ✅ **Good**: Controlled increase
  - ⚠️ **Warning**: Rapid increase = model diverging too much

### How to Identify Diminishing Returns

**Signs of diminishing returns:**
1. **Task reward plateaus**: Accuracy stops increasing for 20+ steps
2. **Loss stops decreasing**: PPO loss flattens out
3. **Advantages near zero**: Model no longer learning from comparisons
4. **Validation accuracy plateaus**: If you have validation metrics

**Example of diminishing returns:**
```
Step 50:  task_reward = 0.25 (25%)
Step 60:  task_reward = 0.28 (28%)
Step 70:  task_reward = 0.29 (29%)
Step 80:  task_reward = 0.29 (29%)  ← Plateau!
Step 90:  task_reward = 0.29 (29%)  ← No improvement
Step 100: task_reward = 0.29 (29%)  ← Diminishing returns
```

## Method 2: Extract Metrics from Log Files

If you can't access WandB, you can extract metrics from the training log:

```bash
# Inside Docker container
cd /workspace/AReaL

# Extract task reward over time
grep -E 'task_reward' ./outputs/grpo/logs/root/gsm8k-grpo-1hour/trial1109_1/trainer.log | \
  grep -E 'avg|mean' | \
  tail -100

# Extract all key metrics
grep -E '(task_reward|ppo_loss|advantages)' ./outputs/grpo/logs/root/gsm8k-grpo-1hour/trial1109_1/trainer.log | \
  tail -50
```

## Method 3: Python Script to Plot Training Curves

Create a script to extract and plot metrics:

```python
#!/usr/bin/env python3
"""Extract and plot training metrics from WandB logs."""

import json
import re
import matplotlib.pyplot as plt
from pathlib import Path

# Path to WandB run directory
wandb_dir = Path("./outputs/grpo/logs/root/gsm8k-grpo-1hour/trial1109_1/wandb")
run_dir = list(wandb_dir.glob("run-*"))[0]

# Read events file
events_file = run_dir / "files" / "wandb-events.jsonl"
if events_file.exists():
    steps = []
    task_rewards = []
    
    with open(events_file) as f:
        for line in f:
            data = json.loads(line)
            if "rollout/task_reward/avg" in data or "grpo_actor/task_reward/avg" in data:
                step = data.get("_step", 0)
                reward = data.get("rollout/task_reward/avg") or data.get("grpo_actor/task_reward/avg")
                if reward is not None:
                    steps.append(step)
                    task_rewards.append(reward * 100)  # Convert to percentage
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(steps, task_rewards, marker='o', label='Task Reward (Accuracy %)')
    plt.xlabel('Training Step')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Curve: Task Reward Over Time')
    plt.grid(True)
    plt.legend()
    plt.savefig('training_curve.png')
    print(f"Plot saved to training_curve.png")
    print(f"Final accuracy: {task_rewards[-1]:.2f}%")
else:
    print("WandB events file not found. Use WandB online dashboard instead.")
```

## Method 4: Check Evaluation Metrics

The evaluation metrics (from test set) are also logged. Check:

1. **During training**: Look for `eval-rollout/task_reward/avg` in WandB
2. **After training**: The final evaluation showed **32.05%** accuracy

### Comparing Training vs Validation

- **Training accuracy**: What the model sees during training
- **Validation accuracy**: What we measured (32.05%)
- **Gap analysis**:
  - If training accuracy >> validation accuracy = **overfitting**
  - If training accuracy ≈ validation accuracy = **good generalization**
  - If training accuracy < validation accuracy = **underfitting** (unlikely)

## Quick Check: Your Current Training

Based on your 1-hour training:

- **Training steps**: ~126 steps (2 epochs × 63 steps/epoch)
- **Final test accuracy**: **32.05%**
- **Base model accuracy**: ~18.35%
- **Improvement**: +13.7 percentage points (+75% relative improvement)

### Analysis

**Good signs:**
- ✅ Significant improvement from base model
- ✅ 32% is a solid result for 1-hour training
- ✅ No signs of overfitting (training and test metrics should be similar)

**To check for diminishing returns:**
1. Look at WandB dashboard for the training curve
2. Check if task_reward was still increasing at the end
3. If it was still increasing, you could train longer
4. If it plateaued, you've reached diminishing returns for this configuration

## Recommendations

### If Task Reward Was Still Increasing:
- ✅ **Train longer**: Increase epochs to 3-5
- ✅ **More data**: Increase dataset size to 1000 samples
- ✅ **More steps**: Should see continued improvement

### If Task Reward Plateaued:
- ⚠️ **Diminishing returns reached** for current config
- ✅ **Try different hyperparameters**: Learning rate, reward scaling
- ✅ **Try different algorithms**: Dr.GRPO, DAPO
- ✅ **More data**: Full dataset (7,473 samples) might help

## Next Steps

1. **Access WandB**: Go to https://wandb.ai and check your project
2. **View training curve**: Look at `rollout/task_reward/avg` over steps
3. **Analyze trend**: Is it still increasing or has it plateaued?
4. **Decide**: Based on the curve, decide if more training is worth it

The training curve will give you the definitive answer about whether you've reached diminishing returns!

