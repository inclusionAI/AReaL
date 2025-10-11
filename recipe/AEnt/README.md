## Overview

Last updated: Oct 11,2025
Doc Author: Han Shen

This is an AReaL-based implementation of [AEnt](https://www.arxiv.org/pdf/2509.03493), a clamped entropy regularization method for LLM-RL algorithms.

Entropy regularization has been a successful method for robotic and games RL, while it offers weak gains for LLM RL. It is argued in the [paper](https://www.arxiv.org/pdf/2509.03493) that entropy regularization suffers from LLM tasks' sparse optimality and the immense response set. 
>One can observe this effect in a toy run on a synthetic MDP below, where as the number of optimal actions decrease (thus sparsity increases), entropy regularization no longer has an advantage over no regularization. The method to be proposed is more robust to this issue.
><div align="left">
  <img src="https://github.com/hanshen95/hanshen95.github.io/blob/master/images/toy_demo.png?raw=true" alt="issue" style="width: 96%; height: auto;">
</div>

To address this issue, AEnt utilizes a clamped entropy regularization paired with adaptively adjusted coefficient. It is observed that AEnt achieves larger gains on multiple benchmarks when tested on different models and training datasets.
> An example run on DeepSeek-R1-distilled-Qwen-1.5b on 40k verifiable samples from Openr1-math dataset
><div align="left">
  <img src="https://github.com/hanshen95/hanshen95.github.io/blob/master/images/score_demo.png?raw=true" alt="issue" style="width: 96%; height: auto;">
</div>


## Configuration

### Clamped entropy 

Related configs:

```yaml
actor:
    aent:
        entropy_coeff: 0.0002
        entropy_clamp: 0.4
```

`entropy_coeff` weighs the entropy regularization and `entropy_clamp` specifies value clamping percentage p in the paper.

Most related code in `functional.py`:

```python
def clamped_softmax_entropy(logits: torch.Tensor, entropy_clamp: float):
    """Assuming softmax policy, calculate entropy with token space clamping."""
    logits_cpu = logits.cpu().detach()
    # compute token space clamping mask
    with torch.no_grad():   
        k = int(logits_cpu.size(-1)*entropy_clamp)
        _, rm_indices = torch.topk(logits_cpu,k=k,dim=-1,largest=False)
        row_indices = torch.arange(logits_cpu.size(0)).unsqueeze(1)
        rm_mask = torch.zeros_like(logits_cpu,dtype=torch.bool)
        rm_mask[row_indices,rm_indices]=True
        del logits_cpu, row_indices, rm_indices
    rm_mask = rm_mask.to(logits.device)
    clamped_logits = logits.masked_fill(rm_mask, -torch.inf)
    clamped_logprobs = F.log_softmax(clamped_logits, dim=-1)
    clamped_logprobs = torch.where(rm_mask, 0., clamped_logprobs)
    del rm_mask
    torch.cuda.empty_cache()
    clamped_probs = F.softmax(clamped_logits, dim=-1)
    clamped_entropy = -torch.sum(clamped_probs * clamped_logprobs, dim=-1)
    return clamped_entropy
```
which is used in `actor.py`:
```python
loss = ppo_loss - entropy_coeff * clamped_entropy_loss
```

### Adaptive coefficient

Related configs:

```yaml
actor:
    aent:
        adaptive_coeff: False
        # following params are disabled if adaptive_coeff==False
        entropy_high: 0.23
        entropy_low: 0.05
        coeff_lr: 0.001
        coeff_box_high: 0.003
        coeff_box_low: 1e-5
        warmup_steps: 29
```

`adaptive_coeff` enables/disables coefficient update during training, `entropy_low/high` sets the lower/upper tolerance of the clamped entropy, `coeff_box_high/low` sets the bounding interval of the coefficient. The coeffcient will start updating after `warmup_steps` with a learning rate of `coeff_lr`.
Though there are many hyper-params here, we find that the algorithm is mostly just sensitive to `entropy_high/low`.

Most related code in `actor.py`:

```python
if self.adaptive_coeff and global_step > self.warmup_steps:
            entropy = sum(ent_trace)/len(ent_trace)
            self.entropy_coeff -= self.coeff_lr*(min(0,entropy-self.entropy_low)+max(0,entropy-self.entropy_high))
            self.entropy_coeff = min(max(self.entropy_coeff, self.coeff_box_low), self.coeff_box_high)
```

## Toy example

We recommend to change the parameter within the configuration file
(i.e.gsm8k_aent_grpo.yaml).

| Backend   | CMD                                                                                                                                            |
| --------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| **local** | `python3 -m areal.launcher.local recipe/AEnt/gsm8k_aent_grpo.py --config recipe/AEnt/configs/gsm8k_aent_grpo.yaml --<other_args_to_overwrite>` |


## Citation
```bibtex
@article{shen2025entropy,
  title={On Entropy Control in LLM-RL Algorithms},
  author={Shen, Han},
  journal={arXiv preprint arXiv:2509.03493},
  year={2025}
}
```

<!-- 
## Baseline

Fine-tuning Qwen2.5-1.5b-Instruct on GSM8K. Results are last verified for the implementation based on **commit #298**.

| lr       | weight decay | group size | entropy coeff | entropy clamp  | max test score |
| -------- | ------------ | ---------- | --------------| ---------------| ---------------|
| 1.50E-05 | 0.           | 8          | 0.0020        | 0.40           | **0.79971**    |
| 1.50E-05 | 0.           | 8          | 0.0025        | 0.20           | 0.79863        |


The baseline result achieved by unregularized GRPO as reported in `examples/math/README.md'
| lr       | weight decay | group size | max task_reward |
| -------- | ------------ | ---------- | --------------- |
| 1.70E-05 | 0.017        | 4          | **0.79570**     |
| 1.30E-05 | 0.015        | 8          | 0.79355         |
| 1.50E-05 | 0.01         | 4          | 0.79043         |
| 1.50E-05 | 0.02         | 4          | 0.78984         |
| 1.00E-05 | 0.02         | 4          | 0.78311         |
| 1.00E-05 | 0.01         | 8          | 0.78066         |


- Devices: 8 Nvidia A100 GPUs
- Optimizer: Adam
- LR Scheduler: Constant
- Gradient Clipping: 1.0
- Max_new_tokens: 1024
- Max_head_offpolicyness: 2
- Training Time: ~70 minutes -->
