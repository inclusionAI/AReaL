# On-Policy Distillation

## Overview 

On-policy distillation trains the student using teacher guidance on trajectories sampled from its own policy, reducing distribution mismatch and improving stability. Combined with reinforcement learning, it lets the student **imitate the teacher while exploring simultaneously**.

**AReaL** previously supported RL for post-training. With this implementation, it now also supports **on-policy knowledge distillation** and the **combined KDRL framework**, enabling the student to learn from a teacher while exploring via RL on the same on-policy trajectories, improving both efficiency and stability.

## Idea


During on-policy knowledge distillation, the student policy $π_θ$ is trained to mimic a teacher $π_T$ while sampling trajectories from its **own current policy**, reducing exposure bias compared to off-policy supervised fine-tuning.

Instead of the standard forward KL ($D_{KL}(π_T || π_θ)$), we use **reverse KL (RKL)**:


$arg$ $min_θ$ $D_{KL}(π_θ || π_T)$ = $arg\ max_θ$ $E_{q \sim Q, o \sim π_θ(. | q)}$ $[log\frac{π_T(o|q)}{π_θ(o|q)}]$

This encourages the student to prefer trajectories the teacher assigns higher probability to and suppress unlikely ones.


## Running the example 

Need to add teacher configuration to your yaml:

```yaml
teacher:
  allocation_mode: d1p1t4
  rl_loss_weight: 1.0
  distill_loss_weight: 0.005
  experiment_name: ${experiment_name}
  trial_name: ${trial_name}
  path: Qwen/Qwen3-32B
  init_from_scratch: false
  disable_dropout: true
  dtype: ${actor.dtype}
  mb_spec:
    max_tokens_per_mb: 10240
  optimizer: null
  scheduling_spec: ${actor.scheduling_spec}
```

```bash
python3 examples/math/gsm8k_rl.py --config examples/distillation/gsm8k_grpo_distill.yaml scheduler.type=local experiment_name=gsm8k-grpo-distillation trial_name=trial0
```

## Result

On-policy knowledge distillation + RL reward plot for Qwen2.5-14B-Instruct (teacher) and Qwen3-0.6B (student), trained using FSDP and vLLM.

![alt text](reward_curve.png)

## References

[KDRL](https://arxiv.org/pdf/2506.02208): Xu H, Zhu Q, Deng H, Li J, Hou L, Wang Y, Shang L, Xu R, Mi F. Kdrl: Post-training reasoning llms via unified knowledge distillation and reinforcement learning.