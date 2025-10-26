# Finding Training Metrics in W&B

## Where to Look in W&B Dashboard

When you're in the W&B dashboard (https://wandb.ai/tong-zhao-georgia-institute-of-technology/areal-gsm8k-mac), here's where to find training progress:

### Main Metrics Tab

1. **Go to your run**: Click on the run (e.g., "summer-dust-4")
2. **Look at the "Metrics" tab** (usually on the left sidebar)
3. **You should see these charts**:
   - `train/loss` - Training loss over time
   - `train/learning_rate` - Learning rate schedule
   - System metrics (CPU, memory, etc.)

### Charts to Look For

**For training progress, look for:**

- **train/loss** - This decreases as training improves
- **global_step** - Shows how many steps have been completed
- **train/learning_rate** - Shows the learning rate

### If Charts Don't Show

The metrics are logged **every 10 global steps**, so if training is very slow, you might not see them yet.

Check the raw log:
```bash
tail -f /tmp/training_cpu.log
```

## Current Status

⚠️ **Training is VERY SLOW on CPU** - 409 seconds per iteration!

At this rate, it would take **~56 hours** to complete.

## Better Solution

Let me create a faster training setup for you with better parameters.

