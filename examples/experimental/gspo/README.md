# Group Sequence Policy Optimization (GSPO)

This directory contains an example implementation of GSPO for training language models
on the GSM8K mathematical reasoning task.

## What is GSPO?

GSPO (Group Sequence Policy Optimization) is a variant of PPO that replaces per-token
probability ratio computation with sequence-level geometric mean of per-token
probability ratios.

### Key Difference from Vanilla PPO

- **Vanilla PPO** (`importance_sampling_level: token`): Computes a separate importance
  sampling ratio for each token: `ratio[i] = exp(logprob[i] - old_logprob[i])`
- **GSPO** (`importance_sampling_level: sequence`): Computes one ratio per sequence as
  the geometric mean: `ratio = exp(mean(logprob - old_logprob))`

This sequence-level ratio is then applied uniformly to all tokens within each sequence,
which can lead to more stable policy updates when optimizing for sequence-level rewards.

## Usage

To enable GSPO, set `importance_sampling_level: sequence` in the actor configuration:

```yaml
actor:
  importance_sampling_level: sequence  # 'token' for standard PPO, 'sequence' for GSPO
  # ... other configurations
```

## Running the Example

```bash
# Run GSPO training on GSM8K
python3 -m areal.launcher.local examples/experimental/gspo/gsm8k_gspo.py --config examples/experimental/gspo/gsm8k_gspo.yaml
```

## Configuration

The example configuration (`gsm8k_gspo.yaml`) specifically includes:

- **importance_sampling_level: sequence** - Enables GSPO algorithm (sequence-level
  importance sampling)

Therefore, you may also plug in such sequence level importance sampling to any other
XXPO algorithms by simply adding the following line after the running command:

```
+actor.importance_sampling_level=sequence
```

For example, you can plug it into GRPO like this:

```
python3 -m areal.launcher.local examples/math/gsm8k_grpo.py --config examples/math/gsm8k_grpo.yaml +actor.importance_sampling_level=sequence
```

Please note the plus sign (`+`) is mandatory when the key `importance_sampling_level` is
not in the yaml file.

## Implementation Details

The GSPO implementation is located in `areal/utils/functional.py`. The algorithm:

1. Computes log probability ratios: `log_ratio = logprobs - proximal_logprobs`
1. Computes mean log ratio per sequence:
   `seq_log_ratio_mean = mean(log_ratio, dim=sequence_length)`
1. Applies geometric mean: `ratio = exp(seq_log_ratio_mean)`
1. Broadcasts the sequence-level ratio to all tokens in each sequence

## When to Use GSPO

GSPO may be beneficial when:

- Training with sequence-level rewards (e.g., task success/failure)
- Dealing with high variance in per-token gradients
- Optimizing for long-horizon tasks where token-level credit assignment is difficult

## Comparison with DAPO

While DAPO (Dynamic Adaptive Policy Optimization) focuses on dynamic sampling and reward
filtering, GSPO focuses on the computation of importance sampling ratios. These
approaches are complementary and can be used together.
