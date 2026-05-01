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
        sampler_cfg = data_config.get("sampler", {})
        self.rollout_n = sampler_cfg.get("rollout_n", 1)
        self.tau = sampler_cfg.get("tau", 1.0)
        self.epoch_decay_start = sampler_cfg.get("epoch_decay_start", 2.0)
        self.epoch_decay_slope = sampler_cfg.get("epoch_decay_slope", 0.3)
        self.epoch_decay_floor = sampler_cfg.get("epoch_decay_floor", 0.1)
        self.batch_size = data_config.get("train_batch_size", 64)
        self.total_training_steps = sampler_cfg.get("total_training_steps", 200)
        self.ema_alpha = sampler_cfg.get("ema_alpha", 0.5)
        self.l_hat_update_ratio = sampler_cfg.get("l_hat_update_ratio", 0.025)

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
        self.acc_buffer: dict[str, list[float]] = {t: [] for t in self.task_types}
        self.L_hat_ema: dict[str, float] = {t: 0.0 for t in self.task_types}
        self.acc_ema: dict[str, float] = {t: 0.5 for t in self.task_types}
        self.acc_ema_prev: dict[str, float] = {t: 0.5 for t in self.task_types}
        self.n_samples: dict[str, int] = {t: 0 for t in self.task_types}
        self.n_total = 0
        self._step_count = 0

        K = max(len(self.task_types), 1)
        self.base_probs: dict[str, float] = {t: 1.0 / K for t in self.task_types}
        self.sampling_probs: dict[str, float] = dict(self.base_probs)
        self._rng = np.random.default_rng(seed=42)

        self._index_pools: dict[str, list[int]] = {}
        for t in self.task_types:
            pool = list(self.task2indices[t])
            self._rng.shuffle(pool)
            self._index_pools[t] = pool
        self._pool_cursors: dict[str, int] = {t: 0 for t in self.task_types}

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

        acc_values = batch.non_tensor_batch.get("accuracy", None)

        for i, task in enumerate(task_labels):
            task_str = str(task)
            if task_str in self.period_buffer:
                self.period_buffer[task_str].append(float(episode_adv_abs[i]))
                if acc_values is not None:
                    self.acc_buffer[task_str].append(float(acc_values[i]))
                self.n_samples[task_str] += 1
                self.n_total += 1

        self._step_count += 1

        if self._step_count % self.l_hat_update_interval == 0:
            self._update_ema()

        self._recompute_probs()

    def _update_ema(self):
        for d_j in self.task_types:
            buf = self.period_buffer[d_j]
            if buf:
                period_mean = float(np.mean(buf))
                self.L_hat_ema[d_j] = self.ema_alpha * period_mean + (1 - self.ema_alpha) * self.L_hat_ema[d_j]
            self.period_buffer[d_j] = []

            acc_buf = self.acc_buffer[d_j]
            if acc_buf:
                acc_mean = float(np.mean(acc_buf))
                self.acc_ema_prev[d_j] = self.acc_ema[d_j]
                self.acc_ema[d_j] = self.ema_alpha * acc_mean + (1 - self.ema_alpha) * self.acc_ema[d_j]
            self.acc_buffer[d_j] = []

    def _recompute_probs(self):
        scores = []
        for d_j in self.task_types:
            explore = math.sqrt(2.0 * math.log(self.n_total + 1) / (self.n_samples[d_j] + 1))
            epoch = self.n_samples[d_j] / (max(self.dataset_sizes[d_j], 1) * self.rollout_n)
            decay = max(self.epoch_decay_floor, 1.0 - self.epoch_decay_slope * max(0.0, epoch - self.epoch_decay_start))
            exploit = 1.0 - 0.5 * self.acc_ema[d_j]
            scores.append(self.base_probs[d_j] * (exploit + explore) * decay)

        scores_t = torch.tensor(scores, dtype=torch.float32)
        probs = F.softmax(scores_t / self.tau, dim=0).numpy()
        self.sampling_probs = {t: float(probs[i]) for i, t in enumerate(self.task_types)}

    def get_metrics(self) -> dict:
        metrics: dict[str, float] = {}
        for d_j in self.task_types:
            metrics[f"at_gigpo/{d_j}/acc_ema"] = self.acc_ema[d_j]
            metrics[f"at_gigpo/{d_j}/acc_raw"] = float(np.mean(self.acc_buffer[d_j])) if self.acc_buffer[d_j] else self.acc_ema[d_j]
            metrics[f"at_gigpo/{d_j}/base_prob"] = self.base_probs[d_j]
            metrics[f"at_gigpo/{d_j}/sampling_prob"] = self.sampling_probs[d_j]
            metrics[f"at_gigpo/{d_j}/epoch_count"] = self.n_samples[d_j] / (max(self.dataset_sizes[d_j], 1) * self.rollout_n)
            metrics[f"at_gigpo/{d_j}/period_buffer_size"] = len(self.period_buffer[d_j])

        probs = list(self.sampling_probs.values())
        metrics["at_gigpo/task_weight_entropy"] = -sum(p * math.log(p + 1e-8) for p in probs)
        return metrics

    def _draw_from_pool(self, task: str, n: int) -> list[int]:
        pool = self._index_pools[task]
        cursor = self._pool_cursors[task]
        result = []
        remaining = n
        while remaining > 0:
            available = len(pool) - cursor
            if available <= 0:
                self._rng.shuffle(pool)
                cursor = 0
            take = min(remaining, len(pool) - cursor)
            result.extend(pool[cursor:cursor + take])
            cursor += take
            remaining -= take
        self._pool_cursors[task] = cursor
        return result

    def __iter__(self):
        probs = np.array([self.sampling_probs[t] for t in self.task_types])
        counts = self._rng.multinomial(self.batch_size, probs)

        indices: list[int] = []
        for i, task in enumerate(self.task_types):
            n_k = int(counts[i])
            if n_k == 0:
                continue
            indices.extend(self._draw_from_pool(task, n_k))

        self._rng.shuffle(indices)
        return iter(indices)

    def state_dict(self) -> dict:
        return {
            "L_hat_ema": dict(self.L_hat_ema),
            "acc_ema": dict(self.acc_ema),
            "acc_ema_prev": dict(self.acc_ema_prev),
            "n_samples": dict(self.n_samples),
            "n_total": self.n_total,
            "_step_count": self._step_count,
            "sampling_probs": dict(self.sampling_probs),
            "_pool_cursors": dict(self._pool_cursors),
            "_index_pools": {t: list(p) for t, p in self._index_pools.items()},
            "_rng_state": self._rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict) -> None:
        for t in self.task_types:
            if t in state.get("L_hat_ema", {}):
                self.L_hat_ema[t] = state["L_hat_ema"][t]
            if t in state.get("acc_ema", {}):
                self.acc_ema[t] = state["acc_ema"][t]
            if t in state.get("acc_ema_prev", {}):
                self.acc_ema_prev[t] = state["acc_ema_prev"][t]
            if t in state.get("n_samples", {}):
                self.n_samples[t] = state["n_samples"][t]
            if t in state.get("sampling_probs", {}):
                self.sampling_probs[t] = state["sampling_probs"][t]
            if t in state.get("_pool_cursors", {}):
                self._pool_cursors[t] = state["_pool_cursors"][t]
            if t in state.get("_index_pools", {}):
                self._index_pools[t] = state["_index_pools"][t]
        self.n_total = state.get("n_total", self.n_total)
        self._step_count = state.get("_step_count", self._step_count)
        if "_rng_state" in state:
            self._rng.bit_generator.state = state["_rng_state"]

    def __len__(self):
        return self.batch_size
