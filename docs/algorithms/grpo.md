# Group Relative Policy Optimization (GRPO)

Last updated: Sep 11,2025

Author: [Ziyi ZENG](https://github.com/ZiyiTsang)

![alt text](../figures/grpo.png)

Group Relative Policy Optimization (GRPO), introduced in DeepSeekMath (Shao et al.,
2024), is an RL method that removes the need for a value function (critic). Instead, it
estimates advantage by normalizing rewards within a group of sampled responses for the
same prompt. This normalization emphasizes differences between candidate outputs,
preserving the reliability of the gradient signal even when rewards are sparse.

The overall surrogate objective is:

$$ J\_{\\text{GRPO}}(\\theta) = \\mathbb{E}{\\substack{q \\sim P(Q), \\ {o_i}{i=1}^G
\\sim \\pi\_{\\theta\_{\\text{old}}}(o|q)}} \\left\[ \\frac{1}{G} \\sum\_{i=1}^G
\\sum\_{t=1}^{|o_i|} \\min\\left(r\_{i,t}(\\theta) \\hat{A}{i,t},
\\text{clip}\\left(r{i,t}(\\theta), 1-\\epsilon, 1+\\epsilon\\right)
\\hat{A}{i,t}\\right) - \\beta D{\\mathrm{KL}}\\left\[\\pi\_\\theta |
\\pi\_{\\text{ref}}\\right\] \\right\] $$

where:

$$ r\_{i,t}(\\theta) = \\frac{\\pi\_\\theta(o\_{i,t} \\mid q,
o\_{i,\<t})}{\\pi\_{\\theta\_{\\text{old}}}(o\_{i,t} \\mid q, o\_{i,\<t})} $$

For more details:

- AReal Detail: [Paper of AReal](https://arxiv.org/abs/2505.24298)

- GRPO Detail: [Paper of DeepSeekMath](https://arxiv.org/pdf/2402.03300)

## Algorithm Core Parameters

- `actor.group_size`: The number of groups to divide the sampled responses into.
- `actor.path`: The path to the actor model.
- `ref.path`: The path to the reference model (if using a reference model).
- `kl_ctl`: The coefficient for the KL divergence term in the objective.
- `total_train_epochs`: The number of epochs to train the model for.
- `optimizer.lr`: The learning rate for the optimizer.

## Example Usage

We recommend to change the parameter within the configuration file
(i.e.gsm8k_grpo.yaml).

| Backend   | CMD                                                                                                                              |
| --------- | -------------------------------------------------------------------------------------------------------------------------------- |
| **local** | `python3 -m areal.launcher.local examples/math/gsm8k_grpo.py --config examples/math/gsm8k_grpo.yaml --<other_args_to_overwrite>` |
| **ray**   | `python3 -m areal.launcher.ray examples/math/gsm8k_grpo.py --config examples/math/gsm8k_grpo.yaml --<other_args_to_overwrite>`   |
| **slurm** | `python3 -m areal.launcher.slurm examples/math/gsm8k_grpo.py --config examples/math/gsm8k_grpo.yaml --<other_args_to_overwrite>` |

## Baselines

We still lack baseline, welcome to contribute!
