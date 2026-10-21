# SPDX-License-Identifier: Apache-2.0

"""Reproduce the observable parts of AReaL dispatch, PPO splitting and FFD.

Exact per-sequence lengths are optional. Summary-only predictions explicitly use
surrogate lengths; they never claim the inferred bins were observed in a trace.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass

from .config import ActorConfig
from .workload import WorkloadStats


def align(value: int, multiple: int) -> int:
    return math.ceil(value / multiple) * multiple


def balanced_partition(lengths: list[int], groups: int) -> list[list[int]]:
    """AReaL balanced_greedy_partition: equal counts, then greedily balance tokens."""
    return [[lengths[i] for i in rank] for rank in _partition_indices(lengths, groups)]


def _partition_indices(lengths: list[int | float], groups: int) -> list[list[int]]:
    if len(lengths) < groups or len(lengths) % groups:
        raise ValueError("Atomic dispatch item count must be >= DP and divisible by DP")
    capacity = len(lengths) // groups
    result = [[] for _ in range(groups)]
    sums = [0] * groups
    for index in sorted(range(len(lengths)), key=lambda i: -lengths[i]):
        selected = min(
            (g for g in range(groups) if len(result[g]) < capacity),
            key=lambda g: sums[g],
        )
        result[selected].append(index)
        sums[selected] += lengths[index]
    return result


def dispatch_workload(
    workload: WorkloadStats, lengths: list[int], dp: int
) -> list[list[int]]:
    if workload.per_dp_sequence_lengths is not None:
        if len(workload.per_dp_sequence_lengths) != dp:
            raise ValueError("per_dp_sequence_lengths must match config DP size")
        return [list(rank) for rank in workload.per_dp_sequence_lengths]
    if workload.trajectory_group_sizes is not None:
        trajectories, offset = [], 0
        for count in workload.trajectory_group_sizes:
            trajectories.append(lengths[offset : offset + count])
            offset += count
        weights = (
            list(workload.trajectory_dispatch_weights)
            if workload.trajectory_dispatch_weights is not None
            else [sum(t) for t in trajectories]
        )
        partitions = _partition_indices(weights, dp)
        return [[n for i in rank for n in trajectories[i]] for rank in partitions]
    if len(lengths) < dp:
        raise ValueError("Actual actor sequences are fewer than DP ranks")
    if len(lengths) % dp == 0:
        return balanced_partition(lengths, dp)
    # Unknown multi-row trajectory grouping: n_seqs%DP need not be zero in the
    # real controller. Use a declared proxy, not a false infeasibility assertion.
    ranks, totals = [[] for _ in range(dp)], [0] * dp
    for length in sorted(lengths, reverse=True):
        chosen = min(range(dp), key=lambda i: totals[i])
        ranks[chosen].append(length)
        totals[chosen] += length
    return ranks


def ffd(
    lengths: list[int], capacity: int, minimum: int, divisor: int = 1
) -> list[list[int]]:
    """Value-only equivalent of utils.seqpack.ffd_allocate (no numpy/numba import)."""
    minimum = max(minimum, divisor)
    if any(n > capacity for n in lengths) or len(lengths) < minimum:
        raise ValueError(
            "Workload exceeds token capacity or has too few sequences for minimum microbatches"
        )
    while True:
        bins, ordered = [], []
        for item in sorted(range(len(lengths)), key=lambda i: -lengths[i]):
            value = lengths[item]
            if len(ordered) < minimum or ordered[0][0] + value > capacity:
                index = len(bins)
                bins.append([item])
                bisect.insort(ordered, (value, index))
            else:
                stop = bisect.bisect_right(ordered, (capacity - value, len(lengths)))
                selected = min(range(stop), key=lambda j: len(bins[ordered[j][1]]))
                total, index = ordered.pop(selected)
                bins[index].append(item)
                bisect.insort(ordered, (total + value, index))
        if len(bins) % divisor == 0:
            # utils.data.allocate_balanced_mbs restores original item ordering.
            return [
                [lengths[i] for i in group]
                for group in sorted(sorted(group) for group in bins)
            ]
        minimum += divisor - minimum % divisor
        if len(lengths) < minimum:
            raise ValueError("Cannot satisfy microbatch divisor with this workload")


@dataclass(frozen=True)
class Microbatch:
    real_tokens: int
    compute_tokens: int
    attention_length_squared: int
    padding_tokens: int


@dataclass(frozen=True)
class PackingPlan:
    logprob: tuple[tuple[Microbatch, ...], ...]  # DP ranks -> synchronized mbs
    train: tuple[
        tuple[tuple[Microbatch, ...], ...], ...
    ]  # PPO optimizer calls -> DP -> mbs
    exact_lengths: bool
    exact_dispatch: bool
    assumptions: tuple[str, ...]

    @property
    def train_compute_tokens(self) -> int:
        return sum(
            mb.compute_tokens for update in self.train for rank in update for mb in rank
        )


def _pad(lengths: list[int], config: ActorConfig, hidden_size: int) -> Microbatch:
    p = config.parallel
    seq_alignment = (
        (p.tp * p.cp * 2 if p.cp > 1 else p.tp) if config.backend == "megatron" else 1
    )
    aligned = [align(n, seq_alignment) for n in lengths]
    total = sum(aligned)
    # Shared utils.data uses 256 tokens; Archon additionally derives a 2MiB page.
    page = max((2 << 20) // hidden_size // 2, 1) if config.backend == "archon" else 256
    batch_alignment = (
        math.lcm(page, p.cp * p.tp) if config.backend == "archon" else page
    )
    computed = (
        config.max_tokens_per_mb
        if config.pad_to_maximum
        else align(sum(lengths), batch_alignment)
    )
    if config.backend == "megatron":
        computed = align(max(computed, total), math.lcm(seq_alignment, 256))
    if computed < total or computed % (
        p.cp * (p.tp if config.backend == "archon" else 1)
    ):
        raise ValueError(
            "Aligned packed microbatch cannot fit the configured token limit/layout"
        )
    padding = computed - total
    # THD padding appends ONE real cu_seqlens segment before attention computation.
    return Microbatch(
        sum(lengths), computed, sum(n * n for n in aligned) + padding * padding, padding
    )


def _pack_ranks(
    shards: list[list[int]],
    config: ActorConfig,
    hidden_size: int,
    observed: int | None = None,
):
    divisor = (
        config.parallel.pp
        if config.backend == "megatron" or config.parallel.pp > 1
        else config.n_mbs_divisor
    )
    minimum = max(config.minimum_microbatches, observed or 1)
    while True:
        grouped = [
            ffd(shard, config.max_tokens_per_mb, minimum, divisor) for shard in shards
        ]
        synchronized = max(len(bins) for bins in grouped)
        if observed is not None and synchronized > observed:
            raise ValueError(
                "Observed microbatch count is incompatible with the workload/config"
            )
        if all(len(bins) == synchronized for bins in grouped):
            break
        minimum = synchronized
    return tuple(
        tuple(_pad(group, config, hidden_size) for group in groups)
        for groups in grouped
    )


def plan_workload(
    config: ActorConfig,
    workload: WorkloadStats,
    hidden_size: int,
    *,
    tail: bool = False,
) -> PackingPlan:
    if workload.train_microbatches is not None and config.ppo_n_minibatches != 1:
        raise ValueError(
            "Average num_micro_batches cannot identify counts of multiple PPO minibatches"
        )
    lengths = workload.lengths(tail=tail, capacity=config.max_tokens_per_mb)
    shards = dispatch_workload(workload, lengths, config.parallel.dp)
    if len(shards[0]) % config.granularity:
        raise ValueError("Local sequence count must divide microbatch granularity")
    if config.granularity != 1:
        raise ValueError(
            "Grouped trajectories need a custom packing component; bundled model supports granularity=1"
        )
    logprob = (
        _pack_ranks(shards, config, hidden_size) if config.compute_prox_logp else ()
    )
    # PPO first balances local sequence groups by count, then engine token-packs each.
    parts = [ffd(shard, sum(shard), config.ppo_n_minibatches) for shard in shards]
    if any(len(groups) != config.ppo_n_minibatches for groups in parts):
        raise ValueError("PPO minibatch partition is inconsistent")
    train = tuple(
        _pack_ranks(
            [groups[i] for groups in parts],
            config,
            hidden_size,
            workload.train_microbatches,
        )
        for i in range(config.ppo_n_minibatches)
    )
    assumptions = []
    exact_lengths = (
        workload.sequence_lengths is not None
        or workload.per_dp_sequence_lengths is not None
    )
    exact_dispatch = workload.per_dp_sequence_lengths is not None or (
        workload.trajectory_group_sizes is not None
        and workload.trajectory_dispatch_weights is not None
        and exact_lengths
    )
    if not exact_lengths:
        assumptions.append(
            "Sequence distribution/DP packing inferred from summary statistics; exact lengths are optional"
        )
    if (
        workload.per_dp_sequence_lengths is None
        and workload.trajectory_group_sizes is None
    ):
        assumptions.append(
            "Unknown atomic trajectory grouping: dispatch uses a per-sequence proxy, not exact DP packing/feasibility; multi-turn trajectories can contain multiple rows"
        )
    elif not exact_dispatch:
        assumptions.append(
            "Trajectory dispatch weights approximated by attended tokens; RTensor controller weights may include rectangular padding"
        )
    if workload.max_seq_len is None and not exact_lengths:
        assumptions.append(
            "Unknown maximum sequence length bounded by max_tokens_per_mb"
        )
    if workload.train_microbatches is None:
        assumptions.append(
            "Synchronized engine microbatch counts inferred, not read from rank logs"
        )
    return PackingPlan(
        logprob, train, exact_lengths, exact_dispatch, tuple(assumptions)
    )
