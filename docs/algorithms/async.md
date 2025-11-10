# Asynchronous RL

AReaL natively supports asynchronous RL training, enabling overlapped rollout generation and model training on disaggregated GPUs. This architecture maximizes GPU utilization by running inference and training concurrently.

> **Note:** This guide applies to all algorithms when asynchronous training is enabled (i.e., `rollout.max_head_offpolicyness > 0`).

## Overview

Traditional online RL algorithms assume synchronous execution: the model generates rollouts, trains on them, and repeats. While simple, this approach leaves GPUs idle when rollout is long and does not scale well.

Asynchronous RL breaks this constraint by overlapping rollout generation and training. However, this introduces **off-policyness**: the policy version generating rollouts may lag behind the training version. To maximize inference throughput, AReaL also supports **partial rollouts**, where a single trajectory can be segmented across multiple policy versions.

## Key Techniques

AReaL addresses the aforementioned algorithmic challenges with two complementary techniques:

### 1. Off-Policyness Control

Limit how stale rollouts can be relative to the current training policy:

```yaml
rollout:
  max_head_offpolicyness: 4  # Allow up to 4 version steps behind
```

**Configuration tips:**
- Set to `0` for synchronous RL (useful for debugging or baseline comparisons)
- Higher values increase throughput but may reduce training stability
- Typical range: 2-8 depending on model size and update frequency

### 2. Decoupled PPO Objective

Handle off-policy data with modified loss computation:

```yaml
actor:
  use_decoupled_loss: true     # Enable decoupled PPO objective
  recompute_logprobs: true     # Recompute logprobs during training
```

**Configuration options:**
- `use_decoupled_loss`: When `false`, uses standard PPO/GRPO objectives
- `recompute_logprobs`: When `false`, reuses logprobs from inference backend
  - **Note:** Must be `true` when `use_decoupled_loss` is enabled

## References

For a practical walkthrough of asynchronous training, see our [GSM8K GRPO example](../lite/gsm8k_grpo.md).

For algorithmic details and empirical analysis, refer to the [AReaL paper](https://arxiv.org/pdf/2505.24298).
