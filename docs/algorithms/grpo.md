# Group Relative Policy Optimization (GRPO)

Last updated: Jan 4, 2026

Author: [Ziyi ZENG](https://github.com/ZiyiTsang)

![grpo figure](../figures/grpo.png)

Group Relative Policy Optimization (GRPO), introduced in DeepSeekMath (Shao et al.,
2024), is an RL method that removes the need for a value function (critic). Instead, it
estimates advantage by normalizing rewards within a group of sampled responses for the
same prompt. This normalization emphasizes differences between candidate outputs,
preserving the reliability of the gradient signal even when rewards are sparse.

The overall surrogate objective is:

$$
J_{\text{GRPO}}(\theta) = \mathbb{E}_{\substack{q \sim P(Q),\\ \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(O|q)}} \left[ \frac{1}{G} \sum_{i=1}^G \sum_{t=1}^{|o_i|} \min\left( r_{i,t}(\theta) \hat{A}_{i,t},\ \text{clip}\left( r_{i,t}(\theta),\ 1-\epsilon,\ 1+\epsilon \right) \hat{A}_{i,t} \right) - \beta D_{\mathrm{KL}}\left[ \pi_\theta \middle| \pi_{\text{ref}} \right] \right]
$$

where:
$$
r_{i,t}(\theta) = \frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,<t})},
\hat{A}_{i,t} = \frac{r_i - \text{mean}(\{r_i\}_{i=1}^G)}{\text{std}(\{r_i\}_{i=1}^G)}.
$$


For more details:

- AReal Detail: [Paper of AReal](https://arxiv.org/abs/2505.24298)

- GRPO Detail: [Paper of DeepSeekMath](https://arxiv.org/pdf/2402.03300)

## Core Parameters

- `actor.group_size`: The number of groups to divide the sampled responses into.
- `actor.path`: The path to the actor model.
- `ref.path`: The path to the reference model (if using a reference model).
- `kl_ctl`: The coefficient for the KL divergence term. Default: `0.0` (disabled for GRPO).
- `actor.adv_norm.mean_level`: Level for calculating mean of advantages. Options: `group`, `batch`, or `none`. Default: `batch`.
- `actor.adv_norm.std_level`: Level for calculating std of advantages. Options: `group`, `batch`, or `none`. Default: `batch`. Set to `none` to match Dr.GRPO/industry best practices.
- `total_train_epochs`: The number of epochs to train the model for.
- `optimizer.lr`: The learning rate for the optimizer.

## Example Usage

We recommend to change the parameter within the configuration file
(i.e.gsm8k_grpo.yaml).

| Backend   | CMD                                                                                                                              |
| --------- | -------------------------------------------------------------------------------------------------------------------------------- |
| **local** | `python3 -m areal.launcher.local examples/math/gsm8k_rl.py --config examples/math/gsm8k_grpo.yaml --<other_args_to_overwrite>` |
| **ray**   | `python3 -m areal.launcher.ray examples/math/gsm8k_rl.py --config examples/math/gsm8k_grpo.yaml --<other_args_to_overwrite>`   |
| **slurm** | `python3 -m areal.launcher.slurm examples/math/gsm8k_rl.py --config examples/math/gsm8k_grpo.yaml --<other_args_to_overwrite>` |

## Baselines

We still lack baseline, welcome to contribute!
