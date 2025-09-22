# Group Relative Policy Optimization Done Right (Dr.GRPO)

Last updated: Sep 11, 2025

Doc Author: [Ziyi ZENG](https://github.com/ZiyiTsang)

![Dr.GRPO figure](https://pica.zhimg.com/v2-03d7071a0b1d9e11a697f2fb25a50068_1440w.jpg)

Dr. GRPO is an advanced optimization method introduced to address the limitations of
previous reinforcement learning approaches in enhancing the reasoning capabilities of
large language models (LLMs). It specifically tackles the issue of optimization bias in
Group Relative Policy Optimization (GRPO) that artificially inflates response lengths,
especially for incorrect outputs. By improving token efficiency while preserving
reasoning performance, Dr. GRPO enables minimalist training recipes to achieve
state-of-the-art results, such as 43.3% accuracy on AIME 2024 with a 7B base model.

For more details:

- AReal Detail: [Paper of AReal](https://arxiv.org/abs/2505.24298)

- Dr.GRPO Detail: [Paper of Dr.GRPO](https://arxiv.org/abs/2503.20783)

## Algorithm Core Parameters

We only list the different parameters from GRPO here:


- `actor.adv_norm.mean_level`: The level when calculate the mean of advantage. options:
  `group`,`batch` or `none`. In dr.GRPO, it is set to `group` by default.
- `actor.adv_norm.std_level`: The level when calculate the std of advantage. options:
  `group`,`batch` or `none`. In dr.GRPO, it is set to `none` by default.

## Example Usage

> The algorithm is experimental and may not be stable.

We recommend to change the parameter within the configuration file
(i.e.gsm8k_drgrpo.yaml).

| Backend   | CMD                                                                                                                                  |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| **local** | `python3 -m areal.launcher.local examples/math/gsm8k_drgrpo.py --config examples/math/gsm8k_drgrpo.yaml --<other_args_to_overwrite>` |
| **ray**   | `python3 -m areal.launcher.ray examples/math/gsm8k_drgrpo.py --config examples/math/gsm8k_drgrpo.yaml --<other_args_to_overwrite>`   |
| **slurm** | `python3 -m areal.launcher.slurm examples/math/gsm8k_drgrpo.py --config examples/math/gsm8k_drgrpo.yaml --<other_args_to_overwrite>` |

## Baselines

We still lack baseline, welcome to contribute!
