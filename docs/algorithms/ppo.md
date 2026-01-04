# Proximal Policy Optimization (PPO)

Last updated: Jan 4, 2026

Author: [Ziyi ZENG](https://github.com/ZiyiTsang)

Proximal Policy Optimization (PPO) is a foundational on-policy reinforcement learning algorithm widely used for training large language models. It balances sample efficiency with implementation simplicity by using a clipped surrogate objective that prevents large policy updates.

## Mathematical Formulation

The PPO objective maximizes the clipped surrogate loss:

$$
J_{\text{PPO}}(\theta) = \mathbb{E}_{t} \left[ \min\left( r_{t}(\theta) \hat{A}_{t},\ \text{clip}\left( r_{t}(\theta),\ 1-\epsilon,\ 1+\epsilon \right) \hat{A}_{t} \right) \right]
$$

where:
- $r_{t}(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$ is the probability ratio between new and old policies
- $\hat{A}_{t}$ is the estimated advantage at timestep $t$
- $\epsilon$ is the clipping parameter (typically 0.2 for RL, often larger like 0.4 for LLMs)

The clipping mechanism ensures that policy updates remain within a trust region, preventing catastrophic updates that could destabilize training.



## Core Parameters

- `actor.eps_clip`: PPO clipping parameter (default: varies by algorithm, e.g., 0.4 for GRPO)
- `kl_ctl`: KL regularization coefficient (set to 0.0 for critic-free methods like GRPO)
- `actor.discount`: Discount factor γ for future rewards (default: 1.0)
- `actor.gae_lambda`: GAE lambda parameter for advantage estimation (default: 1.0)
- `actor.adv_norm`: Advantage normalization configuration (mean/std levels)


## References

- Original PPO paper: [Proximal Policy Optimization Algorithms (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
- AReaL implementation: [Paper of AReaL](https://arxiv.org/abs/2505.24298)
