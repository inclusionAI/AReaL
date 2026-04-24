# AT-GiGPO Implementation Plan (v2 — Safety-Reviewed)

**Adaptive-Task Group-in-Group Policy Optimization**

## Problem

Multi-task multi-turn tool-calling RL training with heterogeneous tasks:
- Easy tasks (1 tool call): high initial accuracy, fast convergence
- Hard tasks (multi-step tool calls): low accuracy, sparse rewards, slow learning
- Data imbalance: e.g. 396 cases vs 3000 cases across tasks
- Optimization conflict: easy tasks dominate gradients, hard tasks stagnate

## Core Idea

One advantage signal (from GiGPO), two pathways:
- **Path 1 (Training)**: per-task advantage rescaling → equalize task gradient contributions
- **Path 2 (Curriculum)**: raw |A_episode| statistics → drive adaptive task sampling

No additional modules (DISCO/DUMP absorbed). GiGPO unchanged.

## Architecture

```
                    GiGPO (unchanged)
                   raw A_total per sample
                         │
              ┌──────────┴──────────┐
              │                     │
        Path 1: Training      Path 2: Curriculum
     per-task advantage        track raw |A_episode|
     rescale in fit()          → ATGiGPOSampler.update()
     → gradient update         → next step's sampling P(d_j)
```

## Backward Compatibility Guarantee

**Zero existing function signatures modified.** All changes are:
- Additive (new fields in existing dicts, new files, new config keys)
- Gated behind `config.algorithm.get("at_gigpo", {}).get("enable", False)`
- When AT-GiGPO is disabled, code paths are identical to current behavior

### Safety audit summary

| Component | Approach | Touches existing signatures? |
|---|---|---|
| GiGPO advantage fn | Piggyback on `gigpo_metrics` dict | NO |
| Loss balancing | Rescale `advantages` in `fit()` before actor update | NO |
| `agg_loss()` | Not modified | NO |
| `PolicyLossFn` type | Not modified | NO |
| `compute_policy_loss_*` | Not modified | NO |
| `dp_actor.py` | Not modified | NO |
| `ATGiGPOSampler` | New file, uses existing `AbstractCurriculumSampler` | NO |

## Key Design Decision: Per-Task Advantage Rescaling (NOT Loss Modification)

GiGPO computes advantage within each group (same prompt, G rollouts).
We cannot do per-task advantage normalization (re-normalizing already-normalized values).
We also cannot modify `agg_loss` or `PolicyLossFn` signature (would break all loss functions).

**Solution**: rescale advantage magnitudes per-task in `fit()`, before the batch is sent
to the actor. This is mathematically equivalent to per-task loss normalization but
requires zero changes to the loss computation chain.

```python
# In fit(), after compute_advantage(), before update_actor()
# Equivalent to: loss = (1/K) * sum_k [ mean(loss_b for b in task_k) ]
# Achieved by: advantage_b *= batch_size / (K * n_k)

advantages = batch.batch["advantages"]
task_labels = batch.non_tensor_batch["data_source"]
unique_tasks = list(set(task_labels))
K = len(unique_tasks)
bs = len(task_labels)
for task in unique_tasks:
    mask = np.array([t == task for t in task_labels])
    n_k = mask.sum()
    advantages[mask] *= (bs / (K * n_k))
batch.batch["advantages"] = advantages
```

Why this works: the policy gradient is `∇L = E[-A * ∇log π]`. Scaling `A` by a
per-task factor is equivalent to scaling the per-task loss contribution.
The actor sees rescaled advantages, computes loss normally, and the gradient
automatically reflects the per-task balance.

## Files to Modify

### 1. New File: `verl_tool/trainer/ppo/at_gigpo_sampler.py`

UCB-based curriculum sampler implementing `AbstractCurriculumSampler`.

```python
import math
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict, deque
from collections.abc import Sized

from omegaconf import DictConfig
from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler


class ATGiGPOSampler(AbstractCurriculumSampler):
    """
    Adaptive-Task sampler driven by GiGPO advantage signals.
    Uses UCB scores to balance exploitation (high |A| tasks)
    and exploration (under-sampled tasks).
    """

    def __init__(self, data_source: Sized, data_config: DictConfig):
        self.data_source = data_source
        self.tau = data_config.get("tau", 0.1)
        self.window_size = data_config.get("window_size", 300)
        self.epoch_decay_start = data_config.get("epoch_decay_start", 2.0)

        # Build task -> sample indices mapping from dataset
        self.task2indices = defaultdict(list)
        for i in range(len(data_source)):
            task = data_source[i].get("data_source", "unknown")
            self.task2indices[task].append(i)

        self.task_types = sorted(self.task2indices.keys())
        self.dataset_sizes = {t: len(self.task2indices[t]) for t in self.task_types}

        # Per-task tracking
        self.windows = {t: deque(maxlen=self.window_size) for t in self.task_types}
        self.n_samples = {t: 0 for t in self.task_types}
        self.n_total = 0
        self.P = {t: 1.0 / len(self.task_types) for t in self.task_types}

        self.batch_size = data_config.get("train_batch_size", 64)
        self._rng = np.random.default_rng(seed=42)

    def update(self, batch: DataProto) -> None:
        """Called after each training step. Updates UCB stats from batch."""
        if "episode_advantages" not in batch.batch:
            return  # AT-GiGPO not active or GiGPO not used

        episode_adv = batch.batch["episode_advantages"]  # (bs,) or (bs, seq)
        if episode_adv.dim() > 1:
            episode_adv = episode_adv.sum(dim=-1)  # reduce to per-sample scalar
        episode_adv = episode_adv.abs().cpu().numpy()

        task_labels = batch.non_tensor_batch.get("data_source", None)
        if task_labels is None:
            return

        # Update per-task windows and sample counts
        for i, task in enumerate(task_labels):
            if task in self.windows:
                self.windows[task].append(float(episode_adv[i]))
                self.n_samples[task] += 1
                self.n_total += 1

        self._compute_ucb()

    def _compute_ucb(self):
        scores = []
        for d_j in self.task_types:
            # Exploitation: mean |A_episode| from sliding window
            L_hat = float(np.mean(self.windows[d_j])) if self.windows[d_j] else 0.0

            # Exploration: under-sampled tasks get bonus
            explore = math.sqrt(2 * math.log(self.n_total + 1) / (self.n_samples[d_j] + 1))

            # Epoch protection: decay when small dataset over-sampled
            epoch = self.n_samples[d_j] / max(self.dataset_sizes[d_j], 1)
            decay = max(0.1, 1.0 - 0.3 * max(0.0, epoch - self.epoch_decay_start))

            scores.append((L_hat + explore) * decay)

        # Softmax to get sampling probabilities
        scores_t = torch.tensor(scores, dtype=torch.float32)
        probs = F.softmax(scores_t / self.tau, dim=0).numpy()
        self.P = {t: float(probs[i]) for i, t in enumerate(self.task_types)}

    def get_metrics(self) -> dict:
        """Return per-task metrics for logging."""
        metrics = {}
        for d_j in self.task_types:
            metrics[f"at_gigpo/{d_j}/mean_abs_adv"] = (
                float(np.mean(self.windows[d_j])) if self.windows[d_j] else 0.0
            )
            metrics[f"at_gigpo/{d_j}/sampling_prob"] = self.P[d_j]
            metrics[f"at_gigpo/{d_j}/epoch_count"] = (
                self.n_samples[d_j] / max(self.dataset_sizes[d_j], 1)
            )

        probs = list(self.P.values())
        ent = -sum(p * math.log(p + 1e-8) for p in probs)
        metrics["at_gigpo/task_weight_entropy"] = ent
        return metrics

    def __iter__(self):
        """Yield sample indices according to current P(d_j)."""
        probs = np.array([self.P[t] for t in self.task_types])
        counts = self._rng.multinomial(self.batch_size, probs)

        indices = []
        for i, task in enumerate(self.task_types):
            n_k = int(counts[i])
            pool = self.task2indices[task]
            chosen = self._rng.choice(pool, size=n_k, replace=(n_k > len(pool)))
            indices.extend(chosen.tolist())

        self._rng.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return self.batch_size
```

### 2. Modify: `compute_gigpo_outcome_advantage()` in `core_algos.py` (~line 500)

**Change**: 2 lines added. Store `episode_advantages` in `gigpo_metrics` dict.
**Return signature unchanged** (still 3-tuple).

```python
# BEFORE (line ~506):
gigpo_metrics = {
    "gigpo/groups_total": len(final_groups),
    ...
}
return combined, combined, gigpo_metrics if obs_hashes is not None and turn_boundaries is not None else {}

# AFTER:
gigpo_metrics = {
    "gigpo/groups_total": len(final_groups),
    ...
}
# AT-GiGPO: attach episode-level advantages for curriculum signal
gigpo_metrics["_episode_advantages"] = episode_advantages  # (bs,) tensor, not logged
return combined, combined, gigpo_metrics if obs_hashes is not None and turn_boundaries is not None else {}
```

### 3. Modify: `compute_advantage()` in `ray_trainer.py` (~line 276)

**Change**: 3 lines added after existing advantage computation.
Extract `_episode_advantages` from metrics dict into batch.

```python
# AFTER existing code at ~line 281:
# data.batch["advantages"] = advantages
# data.batch["returns"] = returns

# AT-GiGPO: extract episode advantages for curriculum sampler
if isinstance(gigpo_metrics, dict) and "_episode_advantages" in gigpo_metrics:
    data.batch["episode_advantages"] = gigpo_metrics.pop("_episode_advantages").unsqueeze(-1) * data.batch["response_mask"]
```

### 4. Modify: `fit()` in `ray_trainer.py` (~line 1269, after compute_advantage)

**Change**: ~15 lines added, all gated behind config check.

```python
# Insert AFTER line 1269 (after compute_advantage returns)
# and BEFORE line 1280 (before update_actor)

# === AT-GiGPO: per-task advantage rescaling ===
at_gigpo_cfg = self.config.algorithm.get("at_gigpo", None)
if at_gigpo_cfg is not None and at_gigpo_cfg.get("enable", False):
    if "data_source" in batch.non_tensor_batch:
        advantages = batch.batch["advantages"]
        task_labels = batch.non_tensor_batch["data_source"]
        unique_tasks = list(set(task_labels))
        K = len(unique_tasks)
        bs = len(task_labels)

        for task in unique_tasks:
            mask_np = np.array([t == task for t in task_labels])
            n_k = mask_np.sum()
            if n_k > 0 and K > 0:
                # Scale factor: equivalent to per-task mean then cross-task mean
                scale = bs / (K * n_k)
                mask_tensor = torch.from_numpy(mask_np).to(advantages.device)
                if advantages.dim() == 1:
                    advantages[mask_tensor] *= scale
                else:  # (bs, seq_len)
                    advantages[mask_tensor.unsqueeze(-1).expand_as(advantages)] *= scale

        batch.batch["advantages"] = advantages

        # Log per-task advantage stats
        for task in unique_tasks:
            mask_np = np.array([t == task for t in task_labels])
            task_adv = advantages[torch.from_numpy(mask_np).to(advantages.device)]
            metrics[f"at_gigpo/{task}/mean_abs_adv"] = task_adv.abs().mean().item()
            metrics[f"at_gigpo/{task}/n_samples_in_batch"] = int(mask_np.sum())
```

### 5. Modify: `fit()` in `ray_trainer.py` (~line 1359, after training step)

**Change**: ~5 lines added for sampler metrics logging.

```python
# AFTER existing code at line 1359-1360:
# if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
#     self.train_dataloader.sampler.update(batch=batch)

# AT-GiGPO: log sampler metrics
if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
    if hasattr(self.train_dataloader.sampler, 'get_metrics'):
        metrics.update(self.train_dataloader.sampler.get_metrics())
```

### 6. Config: `verl_tool/trainer/config/`

Add AT-GiGPO config options. All optional with defaults.

```yaml
algorithm:
  adv_estimator: gigpo
  # AT-GiGPO: Adaptive-Task scheduling (all optional)
  at_gigpo:
    enable: false              # master switch; false = pure GiGPO
    tau: 0.1                   # softmax temperature for UCB sampling
    window_size: 300           # sliding window for |A| tracking
    epoch_decay_start: 2.0     # start decaying after N epochs per dataset
```

### 7. Wire sampler in training script

In the training script (e.g. `run_mixed_gigpo_v2_0420_multinode.sh` or equivalent
Python entry), configure the dataloader to use `ATGiGPOSampler` when AT-GiGPO is enabled:

```python
from verl_tool.trainer.ppo.at_gigpo_sampler import ATGiGPOSampler

if config.algorithm.get("at_gigpo", {}).get("enable", False):
    sampler = ATGiGPOSampler(
        data_source=train_dataset,
        data_config=config.algorithm.at_gigpo,
    )
    train_dataloader = DataLoader(train_dataset, batch_sampler=sampler)
```

## Implementation Order

1. **`compute_gigpo_outcome_advantage` piggyback episode_advantages** → 2 lines, zero risk
2. **`compute_advantage` extract episode_advantages** → 3 lines, gated
3. **`ATGiGPOSampler`** → new file, isolated
4. **`fit()` advantage rescaling** → ~15 lines, gated behind config
5. **`fit()` sampler metrics** → ~5 lines, gated
6. **Config + training script wiring** → config yaml + entry script

## What Happens When AT-GiGPO is Disabled

| Step | AT-GiGPO OFF | AT-GiGPO ON |
|---|---|---|
| `compute_gigpo_outcome_advantage` | `_episode_advantages` added to metrics dict but immediately popped | Same |
| `compute_advantage` | `episode_advantages` stored in batch (harmless extra tensor) | Same, consumed by sampler |
| `fit()` rescaling block | `config.get("at_gigpo")` returns None → skipped entirely | Executes rescaling |
| Sampler | Default sampler (uniform) → `AbstractCurriculumSampler` check is False | `ATGiGPOSampler.update()` called |
| Actor update | Normal advantages, normal loss | Rescaled advantages, normal loss |

**Net effect when disabled**: one extra key in `gigpo_metrics` dict that gets popped, one extra
tensor in batch. No functional difference.

## Validation Plan

1. **Smoke test**: run existing GiGPO config unchanged, verify identical metrics
2. **Unit test**: advantage rescaling with known inputs:
   - 2 tasks, 6 samples (4 from task A, 2 from task B)
   - Verify task B samples get 2x advantage scale
3. **Integration test**: 2-task toy setup (easy 1-turn + hard multi-turn), verify:
   - Sampling shifts toward hard task as easy task converges
   - Easy task not forgotten (exploration bonus)
   - Small dataset epoch cap triggers
4. **Ablation**: compare vs uniform sampling + flat loss (current baseline)
