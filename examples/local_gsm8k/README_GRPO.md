# GRPO Training for GSM8K

This directory contains a GRPO (Group Relative Policy Optimization) training script for GSM8K, following the AReaL framework format.

## Overview

GRPO is a reinforcement learning algorithm that:
- Uses **PPO (Proximal Policy Optimization)** for policy updates
- Generates multiple completions per prompt and assigns rewards based on correctness
- Uses **no separate reward model** - rewards are computed directly from the math parser
- Trains the model to maximize the probability of correct answers

## Files

- `train_grpo.py`: Main GRPO training script (follows AReaL format)
- `train_grpo.yaml`: Configuration file for GRPO training
- `train_local_simple.py`: Simple SFT training (for comparison)
- `train_hf_trainer.py`: HuggingFace Trainer SFT (for comparison)

## Prerequisites

**⚠️ Windows Users**: For Windows 11/10, use Docker + WSL2. See [DOCKER_WINDOWS_SETUP.md](DOCKER_WINDOWS_SETUP.md) and [CURSOR_DOCKER_WORKFLOW.md](CURSOR_DOCKER_WORKFLOW.md) for setup instructions.

1. Install AReaL framework dependencies
2. Install SGLang for inference serving (required for rollout):
   ```bash
   pip install sglang[all]
   ```
3. Set up WandB (optional but recommended):
   ```bash
   export WANDB_API_KEY=your_api_key_here
   ```

## Training

### Launch GRPO Training

The training script uses AReaL's local launcher:

```bash
python -m areal.launcher.local train_grpo.py \
    --config train_grpo.yaml \
    experiment_name=gsm8k-grpo-local \
    trial_name=trial0 \
    actor.path=Qwen/Qwen2.5-0.5B-Instruct \
    stats_logger.wandb.mode=online \
    stats_logger.wandb.wandb_api_key=your_api_key_here
```

### Key Configuration Parameters

- `actor.path`: Path to the base model (e.g., `Qwen/Qwen2.5-0.5B-Instruct`)
- `total_train_epochs`: Number of training epochs (default: 5)
- `gconfig.n_samples`: Number of completions to generate per prompt (default: 4)
- `gconfig.max_new_tokens`: Maximum tokens per generation (default: 512)
- `train_dataset.batch_size`: Training batch size (default: 8)
- `actor.lr`: Learning rate (default: 1.70e-5)
- `actor.reward_scaling`: Reward scaling factor (default: 10.0)
- `allocation_mode`: Resource allocation (`sglang.d1p1t1+d1p1t1` for single GPU)

### Windows Compatibility Notes

The script should work on Windows with CUDA, but note:
- SGLang must be properly installed and configured
- The local launcher manages the inference server automatically
- Set `num_workers: 0` in dataset configs for Windows compatibility

## How GRPO Works

1. **Rollout Phase**: For each prompt, generate `n_samples` completions using the current policy
2. **Reward Computation**: Use `math_parser.process_results()` to check if each completion is correct (binary reward: 1 for correct, 0 for incorrect)
3. **Advantage Computation**: Compute advantages using group-relative normalization (comparing within the group of samples for each prompt)
4. **Policy Update**: Apply PPO update to increase probability of correct answers and decrease probability of incorrect ones
5. **Weight Synchronization**: Update inference server weights after each training step

## Differences from SFT

| Aspect | SFT | GRPO |
|--------|-----|------|
| **Loss Function** | Cross-entropy on answer tokens | PPO objective with rewards |
| **Reward Signal** | None (just next-token prediction) | Binary reward (correct/incorrect) |
| **Optimization** | Directly maximize likelihood | Maximize expected reward |
| **Multiple Samples** | No | Yes (generates multiple completions) |
| **Answer Quality** | Learns format | Explicitly optimizes for correctness |

## Expected Performance

GRPO typically achieves higher accuracy than SFT alone because it:
- Directly optimizes for correct answers (not just format matching)
- Learns from multiple attempts per question
- Uses group-relative normalization to better compare samples

## Troubleshooting

### SGLang Installation Issues
If you encounter issues with SGLang:
```bash
pip install --upgrade sglang[all]
```

### Memory Issues
Reduce these parameters in `train_grpo.yaml`:
- `train_dataset.batch_size`: Reduce from 8 to 4 or 2
- `gconfig.n_samples`: Reduce from 4 to 2
- `gconfig.max_new_tokens`: Reduce from 512 to 256
- `mb_spec.max_tokens_per_mb`: Reduce from 5120 to 2560

### CUDA Out of Memory
Ensure you're using the correct CUDA version of PyTorch and that only one GPU is being used (set `CUDA_VISIBLE_DEVICES=0`).

## References

- AReaL Framework: https://inclusionai.github.io/AReaL/
- GRPO Algorithm: See `docs/algorithms/grpo.md`
- Original GRPO example: `examples/math/gsm8k_grpo.py`
