# Full Dataset Multi-Session Training Guide

## Overview

This guide explains how to train on **ALL GSM8K training samples** (~7,473 samples) using your local 4080S GPU, broken down into manageable sessions that can be interrupted and resumed automatically.

## Key Features

✅ **Full Dataset**: Processes all ~7,473 GSM8K training samples (no limits)  
✅ **Multi-Session**: Break training into manageable sessions  
✅ **Auto-Resume**: Automatically resumes from last checkpoint  
✅ **Interruptible**: Can stop/start anytime without losing progress  
✅ **Frequent Checkpoints**: Saves every 50 steps or 30 minutes  

## Training Statistics

- **Total Samples**: ~7,473 (full GSM8K training set)
- **Batch Size**: 8
- **Steps per Epoch**: ~934 steps
- **Total Epochs**: 5
- **Total Steps**: ~4,670 steps
- **Estimated Time**: ~3-5 days (at ~1 step/minute)

## Files Created

1. **`gsm8k_grpo_full.yaml`** - Configuration for full dataset training
2. **`gsm8k_grpo_full.py`** - Training script (no sample limits)
3. **`run_full_training.sh`** - Session management script

## Quick Start

### Step 1: Set Up Environment

```bash
# Inside Docker container (or local environment)
cd /workspace/AReaL

# Set WandB API key (optional)
export WANDB_API_KEY=your-api-key-here

# Verify GPU
nvidia-smi
```

### Step 2: Run Training Session

```bash
# Run a training session
bash examples/docker_gsm8k/run_full_training.sh
```

**That's it!** The script will:
- Automatically detect and load last checkpoint (if exists)
- Continue training from where it left off
- Save checkpoints every 50 steps or 30 minutes
- Can be interrupted (Ctrl+C) and resumed later

### Step 3: Resume Training (After Interruption)

Simply run the same command again:

```bash
bash examples/docker_gsm8k/run_full_training.sh
```

The training will **automatically resume** from the last checkpoint!

## How It Works

### Automatic Recovery

The configuration uses `recover.mode: auto`, which:
1. **Checks for checkpoints** on startup
2. **Automatically loads** the last checkpoint if found
3. **Resumes training** from the exact step where it stopped
4. **No manual intervention** needed

### Frequent Checkpoints

Checkpoints are saved:
- **Every 50 steps** (`saver.freq_steps: 50`)
- **Every 30 minutes** (`saver.freq_secs: 1800`)
- **After each epoch** (`saver.freq_epochs: 1`)

This ensures minimal progress loss if interrupted.

### Session Management

Each "session" is simply:
1. Run the training script
2. Train until you want to stop (or it completes)
3. Stop with Ctrl+C (checkpoint saved automatically)
4. Resume later by running the script again

## Training Progress Tracking

### WandB Dashboard

Monitor progress in real-time:
1. Go to https://wandb.ai
2. Project: `gsm8k-grpo-local`
3. View:
   - Training curves (`grpo_actor/task_reward/avg`)
   - Loss curves
   - GPU utilization
   - Step progress

### Local Logs

Check checkpoint progress:
```bash
# Check checkpoints
ls -lh outputs/grpo/checkpoints/gsm8k-grpo-full-local/trial0/

# Check recovery info
ls -lh outputs/grpo/recover/gsm8k-grpo-full-local/trial0/
```

## Session Examples

### Example 1: Daily Sessions

Train for a few hours each day:

```bash
# Day 1: Start training
bash examples/docker_gsm8k/run_full_training.sh
# Let it run for a few hours, then Ctrl+C

# Day 2: Resume (automatically continues from last checkpoint)
bash examples/docker_gsm8k/run_full_training.sh

# Day 3: Resume again
bash examples/docker_gsm8k/run_full_training.sh
# ... and so on until complete
```

### Example 2: Time-Limited Sessions

Train for a specific duration:

```bash
# Train for 2 hours, then stop
timeout 7200 bash examples/docker_gsm8k/run_full_training.sh

# Resume later
bash examples/docker_gsm8k/run_full_training.sh
```

### Example 3: Step-Limited Sessions

Train for a specific number of steps (requires script modification):

```bash
# Set max steps per session (in script or environment)
export MAX_STEPS_PER_SESSION=100
bash examples/docker_gsm8k/run_full_training.sh
```

## Configuration Options

### Adjust Checkpoint Frequency

Edit `gsm8k_grpo_full.yaml`:

```yaml
saver:
  freq_steps: 50    # Save every N steps (reduce for more frequent saves)
  freq_secs: 1800   # Save every N seconds (reduce for more frequent saves)
  freq_epochs: 1    # Save after each epoch
```

### Adjust Recovery Frequency

```yaml
recover:
  mode: auto        # auto, resume, disabled
  freq_steps: 50    # Check for checkpoints every N steps
  freq_secs: 1800   # Check for checkpoints every N seconds
```

### Change Experiment Name

```bash
export EXPERIMENT_NAME=gsm8k-grpo-full-local-v2
bash examples/docker_gsm8k/run_full_training.sh
```

## Troubleshooting

### Training Not Resuming

**Problem**: Training starts from step 0 instead of resuming.

**Solution**:
1. Check checkpoint exists: `ls outputs/grpo/checkpoints/...`
2. Verify `recover.mode: auto` in config
3. Check experiment_name and trial_name match previous run

### Out of Memory (OOM)

**Problem**: GPU runs out of memory.

**Solution**: Enable gradient checkpointing in config:
```yaml
actor:
  gradient_checkpointing: true  # Reduces memory usage
```

### Slow Training

**Problem**: Training is very slow (~1 step/minute).

**This is normal** for full dataset training. Options:
1. Reduce `max_new_tokens` (512 → 256)
2. Reduce `n_samples` (4 → 2)
3. Use smaller batch size if OOM
4. Be patient - full training takes days!

## Expected Results

### Training Progress

- **Step 0-100**: Initial learning, accuracy ~20-30%
- **Step 100-500**: Rapid improvement, accuracy ~30-40%
- **Step 500-1000**: Continued improvement, accuracy ~40-50%
- **Step 1000-2000**: Slower improvement, accuracy ~50-60%
- **Step 2000+**: Fine-tuning, accuracy ~60-70% (for 0.5B model)

### Final Accuracy

For Qwen 0.5B model:
- **Base model**: ~20-30% accuracy
- **After full training**: ~60-70% accuracy (estimated)
- **Best possible**: ~70-80% (model size limited)

## Best Practices

1. ✅ **Save frequently**: Use frequent checkpoints (every 50 steps)
2. ✅ **Monitor progress**: Check WandB dashboard regularly
3. ✅ **Resume promptly**: Resume training soon after interruption
4. ✅ **Verify checkpoints**: Check that checkpoints are being saved
5. ✅ **Be patient**: Full training takes days, but worth it!

## Next Steps After Training

1. **Evaluate model**: Run test script on full test set
2. **Compare results**: Compare with base model and limited training
3. **Analyze improvements**: Check if accuracy improved beyond 30-32% bottleneck
4. **Fine-tune further**: If needed, continue training with adjusted hyperparameters

## Summary

This multi-session training approach allows you to:
- ✅ Process **all** GSM8K training samples
- ✅ Break training into manageable sessions
- ✅ Automatically resume from checkpoints
- ✅ Handle interruptions gracefully
- ✅ Track progress via WandB

**Just run the script, let it train, stop when needed, and resume later!**

