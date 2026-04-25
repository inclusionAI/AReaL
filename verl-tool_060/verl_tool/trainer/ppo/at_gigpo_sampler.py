import math
from collections import defaultdict
from collections.abc import Sized

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler


class ATGiGPOSampler(AbstractCurriculumSampler):
    """UCB-based adaptive task sampler driven by GiGPO episode-level advantage signals."""

    def __init__(self, data_source: Sized, data_config: DictConfig):
        self.data_source = data_source
        self.tau = data_config.get("tau", 1.0)
        self.epoch_decay_start = data_config.get("epoch_decay_start", 2.0)
        self.batch_size = data_config.get("train_batch_size", 64)
        self.total_training_steps = data_config.get("total_training_steps", 200)
        self.ema_alpha = data_config.get("ema_alpha", 0.5)
        self.l_hat_update_ratio = data_config.get("l_hat_update_ratio", 0.025)

        self.l_hat_update_interval = max(1, int(self.total_training_steps * self.l_hat_update_ratio))

        self.task2indices: dict[str, list[int]] = defaultdict(list)
        if hasattr(data_source, "dataframe") and "data_source" in data_source.dataframe.column_names:
            ds_col = data_source.dataframe["data_source"]
            for i, task in enumerate(ds_col):
                self.task2indices[str(task)].append(i)
        else:
            for i in range(len(data_source)):
                item = data_source[i]
                task = item.get("data_source", "unknown") if isinstance(item, dict) else "unknown"
                self.task2indices[task].append(i)

        self.task_types = sorted(self.task2indices.keys())
        self.dataset_sizes = {t: len(self.task2indices[t]) for t in self.task_types}

        self.period_buffer: dict[str, list[float]] = {t: [] for t in self.task_types}
        self.L_hat_ema: dict[str, float] = {t: 0.0 for t in self.task_types}
        self.n_samples: dict[str, int] = {t: 0 for t in self.task_types}
        self.n_total = 0
        self._step_count = 0

        K = max(len(self.task_types), 1)
        self.sampling_probs: dict[str, float] = {t: 1.0 / K for t in self.task_types}
        self._rng = np.random.default_rng(seed=42)

    def update(self, batch: DataProto) -> None:
        if "episode_advantages" not in batch.batch:
            return

        episode_adv = batch.batch["episode_advantages"]
        if episode_adv.dim() > 1:
            mask = batch.batch.get("response_mask", None)
            if mask is not None:
                episode_adv = (episode_adv * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
            else:
                episode_adv = episode_adv.mean(dim=-1)
        episode_adv_abs = episode_adv.abs().cpu().numpy()

        task_labels = batch.non_tensor_batch.get("data_source", None)
        if task_labels is None:
            return

        for i, task in enumerate(task_labels):
            task_str = str(task)
            if task_str in self.period_buffer:
                self.period_buffer[task_str].append(float(episode_adv_abs[i]))
                self.n_samples[task_str] += 1
                self.n_total += 1

        self._step_count += 1

        if self._step_count % self.l_hat_update_interval == 0:
            self._update_l_hat()

        self._recompute_probs()

    def _update_l_hat(self):
        for d_j in self.task_types:
            buf = self.period_buffer[d_j]
            if buf:
                period_mean = float(np.mean(buf))
                self.L_hat_ema[d_j] = self.ema_alpha * period_mean + (1 - self.ema_alpha) * self.L_hat_ema[d_j]
            self.period_buffer[d_j] = []

    def _recompute_probs(self):
        scores = []
        for d_j in self.task_types:
            explore = math.sqrt(2.0 * math.log(self.n_total + 1) / (self.n_samples[d_j] + 1))
            epoch = self.n_samples[d_j] / max(self.dataset_sizes[d_j], 1)
            decay = max(0.1, 1.0 - 0.3 * max(0.0, epoch - self.epoch_decay_start))
            scores.append((self.L_hat_ema[d_j] + explore) * decay)

        scores_t = torch.tensor(scores, dtype=torch.float32)
        probs = F.softmax(scores_t / self.tau, dim=0).numpy()
        self.sampling_probs = {t: float(probs[i]) for i, t in enumerate(self.task_types)}

    def get_metrics(self) -> dict:
        metrics: dict[str, float] = {}
        for d_j in self.task_types:
            metrics[f"at_gigpo/{d_j}/L_hat_ema"] = self.L_hat_ema[d_j]
            metrics[f"at_gigpo/{d_j}/sampling_prob"] = self.sampling_probs[d_j]
            metrics[f"at_gigpo/{d_j}/epoch_count"] = self.n_samples[d_j] / max(self.dataset_sizes[d_j], 1)
            metrics[f"at_gigpo/{d_j}/period_buffer_size"] = len(self.period_buffer[d_j])

        probs = list(self.sampling_probs.values())
        metrics["at_gigpo/task_weight_entropy"] = -sum(p * math.log(p + 1e-8) for p in probs)
        return metrics

    def __iter__(self):
        probs = np.array([self.sampling_probs[t] for t in self.task_types])
        counts = self._rng.multinomial(self.batch_size, probs)

        indices: list[int] = []
        for i, task in enumerate(self.task_types):
            n_k = int(counts[i])
            if n_k == 0:
                continue
            pool = self.task2indices[task]
            chosen = self._rng.choice(pool, size=n_k, replace=(n_k > len(pool)))
            indices.extend(chosen.tolist())

        self._rng.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return self.batch_size
