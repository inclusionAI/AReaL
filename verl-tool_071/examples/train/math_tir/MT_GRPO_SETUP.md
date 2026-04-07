# MT-GRPO Setup Documentation

This document describes the MT-GRPO (Multi-Turn Group Relative Policy Optimization) implementation in verl-tool and how to use it for training.

## Overview

MT-GRPO is a reinforcement learning algorithm that provides **turn-level credit assignment** for multi-turn interactions. It combines:

1. **Outcome-level rewards**: Based on final answer correctness (placed at the last token)
2. **Turn-level rewards**: Based on intermediate tool interaction success (distributed to tokens before `<result>` tags)

### Key Formula

```
advantage = outcome_advantage + turn_advantage_coef * turn_advantage * before_result_mask
```

Where:
- `outcome_advantage`: GRPO advantage computed from outcome rewards
- `turn_advantage`: GRPO advantage computed from turn rewards
- `turn_advantage_coef`: Coefficient controlling turn-level credit (default: 1.0)
- `before_result_mask`: Mask indicating tokens before `<result>` tag

## Implementation Details

### 1. Core Algorithm Registration

MT-GRPO is registered in [verl/verl/trainer/ppo/core_algos.py](../../../verl/verl/trainer/ppo/core_algos.py):

```python
@register_adv_est(AdvantageEstimator.MT_GRPO)
def compute_mt_grpo_outcome_advantage(
    data: DataProto,
    values: torch.Tensor,
    config: DictConfig,
    turn_reward: Optional[List[float]] = None,
    tool_interact_info: Optional[List] = None,
    turn_advantage_coef: float = 1.0,
    **kwargs
) -> torch.Tensor:
    # ... implementation ...
```

**Key functions:**
- `_mt_grpo_normalize()`: Group normalization by UID (same as GRPO)
- `_compute_turn_reward_from_tool_info()`: Computes turn reward from tool interaction info
  - Formula: `0.3 * (success_ratio) + 0.05 * (num_successes)`
  - Only counts valid actions (code executions)
  - Checks for execution errors in observation
- `_find_result_segment()`: Locates `<result>` tag position in sequences
- `compute_mt_grpo_outcome_advantage()`: Main advantage computation

### 2. Reward Manager Support

MT-GRPO requires reward managers to provide `turn_reward` in `reward_extra_info`. Currently supported:

#### A. ToRL Reward Manager

Located at [verl_tool/workers/reward_manager/torl.py](../../../verl_tool/workers/reward_manager/torl.py)

**Turn reward computation:**
```python
def compute_turn_reward_for_torl(tool_interact_info):
    """
    Computes turn reward based on code execution success:
    - 0.3 * (success_ratio) + 0.05 * (num_successes)
    """
```

**Features:**
- Automatically enabled by default (`enable_turn_reward=True`)
- Checks tool_interact_info for valid actions and execution results
- Returns turn_reward through reward_extra_info dict

#### B. GSM8K Code Reward Manager

Located at [verl_tool/workers/reward_manager/gsm8k_code.py](../../../verl_tool/workers/reward_manager/gsm8k_code.py)

Uses same turn reward computation logic as ToRL.

### 3. Parameter Passing

The [verl/verl/trainer/ppo/ray_trainer.py](../../../verl/verl/trainer/ppo/ray_trainer.py) was modified to pass MT-GRPO specific parameters:

```python
# MT-GRPO special handling: pass turn_reward and tool_interact_info
if adv_estimator == "mt_grpo" or adv_estimator == AdvantageEstimator.MT_GRPO:
    if "turn_reward" in data.non_tensor_batch:
        adv_kwargs["turn_reward"] = data.non_tensor_batch["turn_reward"]
    if "tool_interact_info" in data.non_tensor_batch:
        adv_kwargs["tool_interact_info"] = data.non_tensor_batch["tool_interact_info"]
    if config is not None:
        adv_kwargs["turn_advantage_coef"] = config.get("turn_advantage_coef", 1.0)
```

## Usage

### Training with MT-GRPO

Use the provided training script:

```bash
bash examples/train/math_tir/train_1.5b_mt_grpo.sh
```

**Key configuration parameters:**

```bash
# Algorithm selection
rl_alg=mt_grpo  # Use MT-GRPO instead of GRPO
turn_advantage_coef=1.0  # Coefficient for turn-level advantage

# Reward manager
reward_manager=torl  # Must support turn_reward computation

# Agent settings (required for multi-turn interactions)
enable_agent=True
max_turns=10
action_stop_tokens='```output'
```

### Training Script Command

The training is invoked via:

```bash
python3 -m verl_tool.trainer.main_ppo \
    algorithm.adv_estimator=mt_grpo \
    +algorithm.turn_advantage_coef=1.0 \
    algorithm.norm_adv_by_std_in_grpo=False \
    reward_model.reward_manager=torl \
    # ... other parameters ...
```

### Evaluation

For evaluation only (no training):

```bash
# Modify any training script and add:
trainer.val_only=True \
trainer.total_training_steps=0
```

## Consistency with Original Implementation

The MT-GRPO implementation in verl-tool is **100% consistent** with the original [Multi-Turn-RL-Agent](https://github.com/your-org/Multi-Turn-RL-Agent) implementation:

### Core Formulas (Identical)

1. **Turn reward computation:**
   ```python
   0.3 * (successful_executions / total_code_steps) + 0.05 * successful_executions
   ```

2. **Advantage combination:**
   ```python
   advantage = outcome_adv + turn_coef * turn_adv * before_result_mask
   ```

3. **GRPO normalization:**
   ```python
   normalized_reward = (reward - group_mean) / (group_std + epsilon)
   ```

### Adaptations for verl Framework

The main differences are framework-specific adaptations:

1. **Data Interface:**
   - Original: Uses `messages` format from dataset
   - verl-tool: Uses `tool_interact_info` from agent_loop

2. **Registration:**
   - Original: Standalone trainer class
   - verl-tool: Registered via `@register_adv_est()` decorator

3. **Reward Manager:**
   - Original: Inline reward functions in rubrics
   - verl-tool: Separate reward manager classes with `turn_reward` in `reward_extra_info`

## Troubleshooting

### Common Issues

1. **GPU Memory OOM:**
   - Reduce `gpu_memory_utilization` (e.g., from 0.9 to 0.75)
   - Enable `do_offload=True`
   - Reduce `batch_size` or `n` (samples per prompt)

2. **Missing turn_reward:**
   - Ensure reward manager supports MT-GRPO (torl or gsm8k_code)
   - Check that `enable_agent=True` for tool interactions
   - Verify `tool_interact_info` is populated in data

3. **WANDB authentication:**
   - Activate virtual environment: `source .venv/bin/activate`
   - Or set: `export WANDB_API_KEY="your_api_key"`

## File Locations

- **Core algorithm:** [verl/verl/trainer/ppo/core_algos.py:757-1026](../../../verl/verl/trainer/ppo/core_algos.py)
- **Ray trainer modifications:** [verl/verl/trainer/ppo/ray_trainer.py:256-263](../../../verl/verl/trainer/ppo/ray_trainer.py)
- **ToRL reward manager:** [verl_tool/workers/reward_manager/torl.py](../../../verl_tool/workers/reward_manager/torl.py)
- **GSM8K reward manager:** [verl_tool/workers/reward_manager/gsm8k_code.py](../../../verl_tool/workers/reward_manager/gsm8k_code.py)
- **Training script:** [examples/train/math_tir/train_1.5b_mt_grpo.sh](train_1.5b_mt_grpo.sh)

## References

- Original Paper: [Multi-Turn Reinforcement Learning for Agent Tasks](link-to-paper)
- Original Implementation: [Multi-Turn-RL-Agent](https://github.com/your-org/Multi-Turn-RL-Agent)
- verl Framework: [verl Documentation](https://github.com/volcengine/verl)
