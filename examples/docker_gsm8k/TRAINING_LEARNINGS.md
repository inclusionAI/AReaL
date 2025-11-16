# Training Learnings and Best Practices

This document consolidates all key learnings, configurations, and best practices from Docker-based GRPO training on GSM8K.

## Table of Contents

1. [Understanding GRPO](#understanding-grpo)
2. [Training Configurations](#training-configurations)
3. [Answer Extraction and Format Learning](#answer-extraction-and-format-learning)
4. [Single-GPU Setup](#single-gpu-setup)
5. [Common Issues and Fixes](#common-issues-and-fixes)
6. [Recovery Procedures](#recovery-procedures)
7. [Testing and Evaluation](#testing-and-evaluation)
8. [Best Practices](#best-practices)

## Understanding GRPO

### What is GRPO?

**GRPO (Group Relative Policy Optimization)** is a reinforcement learning algorithm designed for tasks with **sparse rewards** - like math problems where you only get a reward at the end (correct or incorrect).

### Key Innovation: No Value Function Needed

Traditional RL methods (like standard PPO) require:
- **Actor**: The model being trained
- **Critic**: A separate value function that estimates expected future rewards

GRPO eliminates the need for a critic by using **group-relative normalization** instead.

### How GRPO Works

1. **Generate Multiple Solutions**: For each math problem, GRPO generates multiple candidate solutions (e.g., 4 different attempts)
2. **Group-Relative Normalization**: Instead of using absolute rewards, GRPO normalizes rewards within each group of solutions
3. **Policy Update**: The model learns to increase probability of tokens in correct solutions and decrease probability in incorrect solutions

**Example**:
```
Group rewards: [1, 1, 0, 0]
Group mean: 0.5
Group std: 0.5

Normalized advantages:
- Solution 1: (1 - 0.5) / 0.5 = +1.0  (boosted)
- Solution 2: (1 - 0.5) / 0.5 = +1.0  (boosted)
- Solution 3: (0 - 0.5) / 0.5 = -1.0  (penalized)
- Solution 4: (0 - 0.5) / 0.5 = -1.0  (penalized)
```

### Why GRPO Helps with Math

1. **Sparse Rewards**: Math problems have binary rewards (correct/incorrect), and GRPO handles this well
2. **No Value Function**: Simpler, faster, more stable than traditional PPO
3. **Relative Quality**: Focuses on which solution is better rather than absolute reward
4. **Works with Small Models**: All model capacity goes to improving the policy

## Training Configurations

### Fast Training (20-30 minutes)

**Purpose**: Quick testing, debugging, demonstration

**Configuration**:
- **Dataset**: 200 samples
- **Epochs**: 1
- **Steps**: ~25
- **n_samples**: 2
- **max_new_tokens**: 256
- **Evaluation**: Disabled

**Usage**:
```bash
python3 -m areal.launcher.local examples/docker_gsm8k/gsm8k_grpo_fast.py \
    --config examples/docker_gsm8k/gsm8k_grpo_fast.yaml \
    experiment_name=gsm8k-grpo-fast \
    trial_name=trial0
```

**Expected Results**: Accuracy improves from ~25% to ~30-35%

### 1-Hour Training

**Purpose**: Balanced training time with meaningful dataset coverage

**Configuration**:
- **Dataset**: 500 samples
- **Epochs**: 2
- **Steps**: ~126
- **n_samples**: 4
- **max_new_tokens**: 512

**Usage**:
```bash
python3 -m areal.launcher.local examples/docker_gsm8k/gsm8k_grpo_1hour.py \
    --config examples/docker_gsm8k/gsm8k_grpo_1hour.yaml \
    experiment_name=gsm8k-grpo-1hour \
    trial_name=trial0
```

**Expected Results**: Accuracy improves from ~18-20% to ~25-30%

### 3-Hour Training

**Purpose**: Extended training with more data for better convergence

**Configuration**:
- **Dataset**: 1000 samples
- **Epochs**: 3
- **Steps**: ~375
- **n_samples**: 4
- **max_new_tokens**: 512

**Usage**:
```bash
python3 -m areal.launcher.local examples/docker_gsm8k/gsm8k_grpo_3hour.py \
    --config examples/docker_gsm8k/gsm8k_grpo_3hour.yaml \
    experiment_name=gsm8k-grpo-3hour \
    trial_name=trial0
```

**Expected Results**: Accuracy improves from ~18-20% to ~35-40%

### Full Dataset Training (5 days)

**Purpose**: Complete training on all GSM8K samples

**Configuration**:
- **Dataset**: 7,473 samples (full training set)
- **Epochs**: 5
- **Steps**: ~4,670
- **n_samples**: 4
- **max_new_tokens**: 512

**Usage**:
```bash
python3 -m areal.launcher.local examples/docker_gsm8k/gsm8k_grpo_full.py \
    --config examples/docker_gsm8k/gsm8k_grpo_full.yaml \
    experiment_name=gsm8k-grpo-full \
    trial_name=trial0
```

**Features**:
- Automatic checkpoint recovery
- Frequent checkpoints (every 50 steps or 30 minutes)
- Can be interrupted and resumed

**Expected Results**: Best accuracy, typically 40-50%+ on GSM8K

## Answer Extraction and Format Learning

### How Answer Extraction Works

The `extract_answer()` function in `areal/reward/math_parser.py` is **format-agnostic** and tries multiple strategies:

1. **`\boxed{answer}` format**: Extracts from `\boxed{9}`
2. **"the answer is" format**: Extracts from `"The answer is 9"`
3. **"final answer is" format**: Extracts from `"Final answer: 9"`
4. **Fallback**: Extracts the last number in the text

**Key Point**: The extractor doesn't require a specific format - it can extract answers from many different formats.

### How Model Learns Format

1. **Prompt Instruction**: Explicitly asks for `\boxed{}` format: `"Please put your final answer within \boxed{}."`
2. **Pre-trained Capability**: Base models (like Qwen) can follow instructions
3. **Flexible Extractor**: Works even if format isn't perfect
4. **RL Learning**: Model learns through rewards which formats work best
5. **Multiple Samples**: 4 solutions per problem increases chance of extractable answer

**The model doesn't need SFT because**:
- ✅ The extractor is flexible (doesn't require perfect format)
- ✅ Pre-trained models can follow instructions
- ✅ RL training teaches the model to produce extractable answers
- ✅ The format is learned implicitly through reward signals

## Single-GPU Setup

### Key Fix: Disk-Based Weight Updates

The training script automatically detects single-GPU setups and uses disk-based weight updates:

```python
# Automatically use disk-based updates for single GPU
if config.cluster.n_gpus_per_node == 1 and allocation_mode.gen.world_size == 1:
    weight_update_meta = WeightUpdateMeta.from_disk(...)
else:
    weight_update_meta = WeightUpdateMeta.from_fsdp_xccl(allocation_mode)
```

This avoids the NCCL "duplicate GPU" error that occurs when training and inference processes share the same GPU.

### Configuration for Single GPU

```yaml
cluster:
  n_gpus_per_node: 1

allocation_mode: sglang.d1p1t1+d1p1t1  # Single GPU allocation

actor:
  path: Qwen/Qwen2.5-0.5B-Instruct  # Small model for single GPU
  batch_size: 8  # Adjusted for single GPU memory
```

## Common Issues and Fixes

### Issue: SGLang Server Crash/Disconnect

**Symptoms**:
- Task reward drops to zero
- Connection refused errors to SGLang server
- Training continues but learns nothing

**Root Causes**:
1. **Out of Memory (OOM)**: Server process killed by system
2. **Memory Leak**: Memory usage gradually increases
3. **Disk I/O Failure**: Weight update fails
4. **SGLang Server Bug**: Segmentation fault or hang

**Solutions**:
```yaml
# Reduce memory usage
sglang:
  mem_fraction_static: 0.6  # Reduced from 0.8

rollout:
  max_concurrent_rollouts: 16  # Reduced from 32
  request_timeout: 7200  # Increased from 3600
  request_retries: 5  # Increased from 3

gconfig:
  max_new_tokens: 256  # Reduced from 512 (if needed)
```

### Issue: Out of Memory

**Symptoms**:
- CUDA out of memory errors
- GPU memory at 100%
- Process killed by system

**Solutions**:
1. Reduce `sglang.mem_fraction_static` to 0.5 or 0.6
2. Enable `actor.gradient_checkpointing: true`
3. Reduce `train_dataset.batch_size` to 4
4. Reduce `gconfig.max_new_tokens` to 256
5. Reduce `rollout.max_concurrent_rollouts` to 16

### Issue: Training Too Slow

**Optimizations**:
1. Reduce `n_samples` from 4 to 2
2. Reduce `max_new_tokens` from 512 to 256
3. Increase `batch_size` if memory allows
4. Disable evaluation during training
5. Use faster training config (fast, 1hour, 3hour)

### Issue: No Improvement in Accuracy

**Possible Causes**:
1. **Insufficient training**: Need more samples/epochs
2. **Learning rate too high**: Model may be unstable
3. **Learning rate too low**: Model learns too slowly
4. **Dataset too small**: Need more diverse examples

**Solutions**:
1. Increase dataset size (500 → 1000 → full)
2. Increase epochs (2 → 3 → 5)
3. Adjust learning rate in config
4. Check WandB for training curves

## Recovery Procedures

### Automatic Recovery

All training scripts support automatic recovery from checkpoints:

```yaml
recover:
  mode: auto  # Automatically resume from last checkpoint
  freq_epochs: 1
  freq_steps: null
  freq_secs: 3600  # Save recovery info every hour
```

**How it works**:
1. Training checks for `recover_checkpoint/` directory on startup
2. If found, automatically loads checkpoint and resumes from exact step
3. No manual intervention needed

### Manual Recovery

If automatic recovery doesn't work:

1. **Find last checkpoint**:
   ```bash
   ls -lt outputs/grpo/checkpoints/gsm8k-grpo-docker/trial0/default/ | head -10
   ```

2. **Verify checkpoint exists**:
   ```bash
   ls outputs/grpo/checkpoints/gsm8k-grpo-docker/trial0/default/epoch*epochstep*globalstep*/
   ```

3. **Resume training** (auto-detects checkpoint):
   ```bash
   python3 -m areal.launcher.local examples/docker_gsm8k/gsm8k_grpo_full.py \
       --config examples/docker_gsm8k/gsm8k_grpo_full.yaml \
       experiment_name=gsm8k-grpo-full \
       trial_name=trial0
   ```

### Checkpoint Locations

**Checkpoints saved to**:
```
outputs/grpo/checkpoints/root/{experiment_name}/{trial_name}/default/epoch{epoch}epochstep{step}globalstep{global_step}/
```

**Recovery info saved to**:
```
outputs/grpo/checkpoints/root/{experiment_name}/{trial_name}/recover_checkpoint/
```

## Testing and Evaluation

### Testing Trained Models

```bash
# Test a trained model
python examples/docker_gsm8k/test_trained_model.py \
    --model-path ./outputs/grpo/checkpoints/root/gsm8k-grpo-docker/trial0/default/epoch0epochstep0globalstep1 \
    --max-samples 10
```

### Accuracy Tracking During Training

**Task reward = Accuracy**:
- `grpo_actor/task_reward/avg` in training logs = accuracy (0.0 = 0%, 1.0 = 100%)
- Tracked automatically in WandB
- Evaluation runs after each epoch

### Viewing Training Curves

1. Go to WandB: https://wandb.ai
2. Open project: `gsm8k-grpo-local`
3. Find your run: `{experiment_name}_{trial_name}_train`
4. Check `grpo_actor/task_reward/avg` curve for accuracy
5. Check `grpo_actor/loss` for training loss

## Best Practices

### Before Training

1. **Set up Docker container** with GPU access
2. **Configure WandB API key** as environment variable
3. **Choose appropriate config** for your time budget
4. **Test with fast config** first to verify setup
5. **Verify GPU access**: `nvidia-smi` inside container

### During Training

1. **Monitor WandB** for training progress
2. **Watch for zero rewards** - indicates SGLang server issues
3. **Check GPU memory** - ensure not approaching limits
4. **Verify checkpoints** - confirm checkpoints are being saved
5. **Monitor training logs** for errors

### After Training

1. **Evaluate model** on test set
2. **Check WandB curves** for convergence
3. **Compare results** with baseline
4. **Save best checkpoint** for deployment

### Configuration Tips

1. **Start with fast config** to verify setup works
2. **Scale up gradually**: fast → 1hour → 3hour → full
3. **Adjust checkpoint frequency** for long runs: `saver.freq_steps: 50`
4. **Enable recovery** for long runs: `recover.mode: auto`
5. **Monitor memory usage** and adjust if needed

### Memory Management

**For single-GPU setups**:
- Use smaller model (0.5B instead of 1.5B+)
- Reduce batch size if OOM occurs
- Enable gradient checkpointing
- Reduce SGLang memory fraction
- Reduce max new tokens

**For multi-GPU setups**:
- Can use larger models
- Larger batch sizes
- More concurrent rollouts

## Summary

✅ **GRPO is ideal for math** - handles sparse rewards, no value function needed  
✅ **Multiple training configs** - fast (20-30 min), 1-hour, 3-hour, full (5 days)  
✅ **Automatic recovery** - resumes from checkpoints seamlessly  
✅ **Format-agnostic extraction** - works with various answer formats  
✅ **Single-GPU support** - automatic disk-based weight updates  
✅ **Monitor via WandB** - track accuracy, loss, and training progress  
✅ **Start small, scale up** - test with fast config, then scale to full training

