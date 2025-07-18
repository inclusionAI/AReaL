import time
from collections import defaultdict
from contextlib import contextmanager
from enum import Enum, auto
from typing import Dict

import torch
import torch.distributed as dist


class ReduceType(Enum):
    AVG = auto()
    SUM = auto()
    MIN = auto()
    MAX = auto()
    SCALAR = auto()


MOE_AUX_LOSSES = {}


class DistributedStatsTracker:
    def __init__(self, name: str = ""):
        self.scope_stack = []
        if name:
            self.scope_stack.append(name.strip("/"))
        self.denominators = {}  # key -> denominator key
        self.reduce_types = {}  # key -> ReduceType

        self.stats = defaultdict(list)

    def scope(self, name):
        """Context manager for hierarchical scoping"""
        return self.Scope(self, name)

    class Scope:
        def __init__(self, tracker, name):
            self.tracker = tracker
            self.name = name.strip("/")

        def __enter__(self):
            self.tracker.scope_stack.append(self.name)
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.tracker.scope_stack.pop()

    def _get_full_key(self, key):
        """Combine scope stack with current key"""
        if not self.scope_stack:
            return key
        return "/".join(self.scope_stack + [key])

    @contextmanager
    def record_timing(self, key):
        start_time = time.perf_counter()
        try:
            yield
        finally:
            # NOTE: timing records are fixed under the "timeperf" scope
            full_key = f"timeperf/{key}"
            self._set_reduce_type(full_key, ReduceType.SCALAR)
            self.stats[full_key].append(time.perf_counter() - start_time)

    def denominator(self, **kwargs):
        for key, value in kwargs.items():
            if not isinstance(value, torch.Tensor) or value.dtype != torch.bool:
                raise ValueError(
                    f"`{key}` must be a pytorch bool tensor: {value.dtype}"
                )
            if value.numel() == 0:
                raise ValueError(f"`{key}` must be non-empty")
            full_key = self._get_full_key(key)
            self._set_reduce_type(full_key, ReduceType.SUM)
            self.stats[full_key].append(value.detach().clone())

    def scalar(self, **kwargs):
        for key, value in kwargs.items():
            full_key = self._get_full_key(key)
            self._set_reduce_type(full_key, ReduceType.SCALAR)
            self.stats[full_key].append(float(value))

    def stat(
        self,
        denominator: str,
        reduce_type: ReduceType | None = None,
        **kwargs,
    ):
        """Record multiple values from a dictionary"""
        for key, value in kwargs.items():
            if not isinstance(value, torch.Tensor) or value.dtype != torch.float:
                raise ValueError(
                    f"`{key}` should be a pytorch float tensor: {value.dtype}"
                )
            if value.numel() == 0:
                raise ValueError(f"`{key}` should be non-empty")
            if reduce_type == ReduceType.SCALAR:
                raise ValueError("Cannot use the scalar reduce type for a tensor")
            full_key = self._get_full_key(key)

            denorm = self._get_full_key(denominator)
            if denorm not in self.stats or not self.stats[denorm]:
                raise ValueError(f"Denominator `{denorm}` does not exist")
            for x, y in zip(self.stats[denorm], self.stats[full_key] + [value]):
                assert x.shape == y.shape, (x.shape, y.shape)
            self.denominators[full_key] = denorm

            if reduce_type is not None:
                self._set_reduce_type(full_key, reduce_type)

            self.stats[full_key].append(value.detach().clone())

    def _set_reduce_type(self, key, reduce_type):
        if not isinstance(reduce_type, ReduceType):
            raise ValueError("reduce_type must be a ReduceType enum")
        self.reduce_types[key] = reduce_type

    def export(self, key=None, reduce_group=None, reset=True) -> Dict[str, float]:
        """Get aggregated statistics"""
        self._amend_moe_losses()
        if reduce_group is None:
            try:
                from realhf.base.constants import data_parallel_group

                reduce_group = data_parallel_group()
            except:
                pass
        if key is not None:
            full_key = self._get_full_key(key)
            result = self._aggregate(full_key, reduce_group)
            if reset:
                if full_key in self.denominators:
                    self.denominators.pop(full_key)
                if full_key in self.reduce_types:
                    self.denominators.pop(full_key)
                self.stats.pop(full_key)
            return result

        results = {}
        for key in list(self.stats.keys()):
            results.update(self._aggregate(key, reduce_group))
        if reset:
            self.denominators = {}
            self.reduce_types = {}
            self.stats = defaultdict(list)
        results = {
            k: v.cpu().item() if torch.is_tensor(v) else v for k, v in results.items()
        }
        return results

    def _amend_moe_losses(self):
        from realhf.base.constants import is_last_pipe_stage, pipe_parallel_group

        global MOE_AUX_LOSSES
        mean_losses = {}
        for k, loss in MOE_AUX_LOSSES.items():
            dist.all_reduce(loss, group=pipe_parallel_group())
            mean_losses[k] = float(loss.mean())  # average over layers
        MOE_AUX_LOSSES.clear()
        if mean_losses and is_last_pipe_stage():
            self.scalar(**mean_losses)

    def _aggregate(self, key, reduce_group):
        if key not in self.stats or not self.stats[key]:
            return {}

        reduce_type = self.reduce_types.get(key, None)

        result = {}
        if reduce_type is None:
            result["/".join([key, "avg"])] = self._avg_of(key, reduce_group)
            result["/".join([key, "min"])] = self._min_of(key, reduce_group)
            result["/".join([key, "max"])] = self._max_of(key, reduce_group)
        elif reduce_type == ReduceType.AVG:
            result[key] = self._avg_of(key, reduce_group)
        elif reduce_type == ReduceType.SUM:
            result[key] = self._sum_of(key, reduce_group)
        elif reduce_type == ReduceType.MIN:
            result[key] = self._min_of(key, reduce_group)
        elif reduce_type == ReduceType.MAX:
            result[key] = self._max_of(key, reduce_group)
        elif reduce_type == ReduceType.SCALAR:
            result[key] = sum(self.stats[key]) / len(self.stats[key])
        else:
            raise ValueError(f"Unknown reduce type: {reduce_type}")

        keys_to_pop = [k for k, v in result.items() if v is None]
        for k in keys_to_pop:
            result.pop(k)
        return result

    def _sum_of(self, key, reduce_group):
        values = self.stats[key]
        if key not in self.denominators:
            x = sum([x.sum() for x in values])
            if reduce_group is not None:
                dist.all_reduce(x, group=reduce_group)
        else:
            denominator = self.denominators[key]
            if denominator not in self.stats:
                raise ValueError(
                    f"Denominator `{denominator}` not set for key `{key}`."
                )
            xs = []
            for v, d in zip(values, self.stats[denominator]):
                xs.append(torch.where(d, v, 0.0).sum())
            x = sum(xs)
            if reduce_group is not None:
                dist.all_reduce(x, group=reduce_group)
        return float(x)

    def _avg_of(self, key, reduce_group):
        values = self.stats[key]
        denominator = self.denominators[key]
        if denominator not in self.stats:
            raise ValueError(f"Denominator `{denominator}` not set for key `{key}`.")
        xs = []
        ds = []
        for v, d in zip(values, self.stats[denominator]):
            xs.append(torch.where(d, v, 0.0).sum())
            ds.append(d.sum())
        x = sum(xs)
        d = sum(ds)
        if reduce_group is not None:
            dist.all_reduce(x, group=reduce_group)
            dist.all_reduce(d, group=reduce_group)
        if d == 0:
            return None
        return x / d

    def _min_of(self, key, reduce_group):
        values = self.stats[key]
        denominator = self.denominators[key]
        if denominator not in self.stats:
            raise ValueError(f"Denominator `{denominator}` not set for key `{key}`.")
        xs = []
        for v, d in zip(values, self.stats[denominator]):
            xs.append(torch.where(d, v, float("inf")).min())
        x = min(xs)
        if reduce_group is not None:
            dist.all_reduce(x, group=reduce_group, op=dist.ReduceOp.MIN)
        if torch.isinf(x):
            return None
        return float(x)

    def _max_of(self, key, reduce_group):
        values = self.stats[key]
        denominator = self.denominators[key]
        if denominator not in self.stats:
            raise ValueError(f"Denominator `{denominator}` not set for key `{key}`.")
        xs = []
        for v, d in zip(values, self.stats[denominator]):
            xs.append(torch.where(d, v, -float("inf")).max())
        x = max(xs)
        if reduce_group is not None:
            dist.all_reduce(x, group=reduce_group, op=dist.ReduceOp.MAX)
        if torch.isinf(x):
            return None
        return float(x)


DEFAULT_TRACKER = DistributedStatsTracker()
stat = DEFAULT_TRACKER.stat
denominator = DEFAULT_TRACKER.denominator
export = DEFAULT_TRACKER.export
scope = DEFAULT_TRACKER.scope
scalar = DEFAULT_TRACKER.scalar
record_timing = DEFAULT_TRACKER.record_timing
