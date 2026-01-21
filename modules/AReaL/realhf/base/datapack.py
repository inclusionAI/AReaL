# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import bisect
import itertools
from typing import Any, List, Tuple, Union

import numba
import numpy as np


def flat2d(arr: List[List[Any]]) -> List[Any]:
    return list(itertools.chain(*arr))


@numba.njit
def partition_balanced(nums: np.ndarray, k: int, min_size: int = 1):
    """Partition an array into k subarrays with a minimum absolute difference
    of sums and minimum subarray size.

    Dynamic programming solution.

    Args:
        nums (np.ndarray): The array to be partitioned.
        k (int): Number of partitions.
        min_size (int): Minimum size of each subarray.

    Returns:
        List[int]: Partition slicing point indices in a list including start and end points.
                   Length equals to k + 1.
    """
    n = len(nums)

    dp = np.full((n + 1, k + 1), dtype=np.int64, fill_value=int(1e10))
    maxval = np.full((n + 1, k + 1), dtype=np.int64, fill_value=-int(1e10))
    minval = np.full((n + 1, k + 1), dtype=np.int64, fill_value=int(1e10))
    prefix_sums = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(nums)), axis=0)
    split = np.zeros((n + 1, k + 1), dtype=np.int64)

    for i in range(n + 1):
        dp[i, 1] = 0
        maxval[i, 1] = prefix_sums[i] - prefix_sums[0]
        minval[i, 1] = prefix_sums[i] - prefix_sums[0]

    for j in range(2, k + 1):
        for i in range(j * min_size, n + 1):
            for x in range(min_size, i - min_size + 1):
                xx = prefix_sums[i] - prefix_sums[x]
                min_diff = max(
                    dp[x, j - 1], maxval[x, j - 1] - xx, xx - minval[x, j - 1]
                )
                dp[i, j] = min(dp[i, j], min_diff)

                if dp[i, j] == min_diff:
                    split[i][j] = x
                    if dp[i, j] == maxval[x, j - 1] - xx:
                        maxval[i, j] = maxval[x, j - 1]
                        minval[i, j] = xx
                    elif dp[i, j] == xx - minval[x, j - 1]:
                        maxval[i, j] = xx
                        minval[i, j] = minval[x, j - 1]
                    else:
                        maxval[i, j] = maxval[x, j - 1]
                        minval[i, j] = minval[x, j - 1]
    res = [n]
    idx = n
    for i in range(k, 0, -1):
        idx = split[idx][i]
        res.append(idx)
    return res[::-1]


def partition_balanced_tuples(
    nums: np.ndarray, k: int, min_size: int = 1
) -> List[Tuple[int, int]]:
    lst = partition_balanced(nums, k, min_size)
    return [(lst[i], lst[i + 1]) for i in range(k)]


def min_abs_diff_partition(
    arr: Union[np.ndarray, List], k: int, min_size: int = 1
) -> List[Tuple[int, int]]:
    err_hint = (
        " Errors should not be reported in this function. It is probably a bug in the dataset code"
        " or too small batch size in pipeline parallel realhf.experiments."
    )

    if isinstance(arr, list):
        arr = np.array(arr)
    if len(arr.shape) > 1:
        raise ValueError(f"The array to be partitioned must be 1D. ({arr})" + err_hint)
    if len(arr) < k:
        raise ValueError(
            f"The array to be partitioned must have length >= k. (array {arr}, k={k})"
            + err_hint
        )
    if len(arr) < k * min_size:
        raise ValueError(
            f"Length of the array to be partitioned must be at least k * min_size ({k} * {min_size}), current length {len(arr)}."
        )
    partitions = partition_balanced_tuples(arr, k, min_size)
    last_end = 0

    err_type = None
    err_msg = f"Lengths to be partitioned: {arr}, k={k}, current partition result {partitions}."
    for start, end in partitions:
        if start != last_end:
            err_type = "not contiguous"
        if end <= start:
            err_type = "empty"
        if err_type:
            raise ValueError(
                f"Partition {start}-{end} is {err_type}. " + err_msg + err_hint
            )
        last_end = end
    return partitions


# @numba.njit
def reorder_to_balanced_batches(
    seqlens: np.ndarray,
    n_seqs_per_batch: int,
) -> Tuple[np.ndarray, int]:
    max_bins = (len(seqlens) + n_seqs_per_batch - 1) // n_seqs_per_batch

    bins = [[] for _ in range(max_bins)]
    bin_sizes = np.zeros(max_bins, dtype=np.int32)
    bin_seqlens = np.zeros(max_bins, dtype=np.int32)
    for i in seqlens.argsort()[::-1]:
        idx = np.where(
            bin_sizes + 1 <= n_seqs_per_batch,
            bin_seqlens,
            np.iinfo(np.int32).max,
        ).argmin()
        bins[idx].append(i)
        bin_sizes[idx] += 1
        bin_seqlens[idx] += seqlens[i]

    assert np.all(bin_sizes <= n_seqs_per_batch), (bin_sizes, n_seqs_per_batch)
    max_diff = 0
    for i in range(max_bins):
        for j in range(i + 1, max_bins):
            max_diff = max(max_diff, abs(bin_seqlens[i] - bin_seqlens[j]))

    reordered_indices = []
    for i in bin_seqlens.argsort()[::-1]:
        reordered_indices.extend(bins[i])
    return np.array(reordered_indices), max_diff


# @numba.njit
def _ffd_allocate(
    values: np.ndarray, capacity: int, min_groups: int
) -> List[List[int]]:
    """A greedy allocation algorithm that partitions a list of numbers
    into k groups, where the summation of each group is less than capacity
    and k >= min_groups. We want to minimize k and make partitions as balanced
    as possible.

    1. Sort the numbers in reverse order.
    2. If the number of groups is less than `min_groups`, create a new group.
    3. For a new number, find all groups with the capacity to hold the new number.
       Put the new number into the group with the smallest size.
    4. Otherwise, create a new group.
    """
    value_indices = np.argsort(-values)
    group_indices: List[List[int]] = []
    group_values: List[Tuple[float, int]] = []
    group_cnt = 0
    for idx in value_indices:
        if (
            len(group_values) < min_groups
            or group_values[0][0] + values[idx] > capacity
        ):
            bisect.insort(group_values, (float(values[idx]), group_cnt))
            group_indices.append([idx])
            group_cnt += 1
        else:
            i = bisect.bisect_right(group_values, (capacity - values[idx], len(values)))
            candidates = [group_values[j][1] for j in range(i)]
            lens = [len(group_indices[g]) for g in candidates]
            j = np.argmin(lens)
            v, group_idx = group_values.pop(j)
            assert group_idx == candidates[j]
            bisect.insort(group_values, (float(values[idx] + v), group_idx))
            group_indices[group_idx].append(idx)
    return group_indices


def ffd_allocate(values: List[int], capacity: int, min_groups: int) -> List[List[int]]:
    if any(v > capacity for v in values):
        raise RuntimeError(f"Values {values} is larger than capacity {capacity}")
    if len(values) < min_groups:
        raise RuntimeError(
            f"Number of values {len(values)} is smaller than min_groups {min_groups}"
        )
    return _ffd_allocate(np.array(values), capacity, min_groups)


if __name__ == "__main__":
    import time

    for i in range(100):
        st = time.monotonic()
        nums = np.random.randint(1024, 8192, size=(100,))
        # k = np.random.randint(2, 20)
        # min_size = np.random.randint(1, len(nums) // k)
        # res = min_abs_diff_partition(nums, k, min_size)
        # assert all(y - x >= min_size for x, y in res)
        max_tokens_per_mb = 163840
        min_n_groups = np.random.randint(1, 8)
        groups = ffd_allocate(nums, max_tokens_per_mb, min_n_groups)
        assert len(groups) >= min_n_groups
        import itertools

        indices = list(itertools.chain(*groups))
        assert len(set(indices)) == len(indices)
        group_percent = [
            sum(nums[i] for i in group) / max_tokens_per_mb for group in groups
        ]

        print(
            len(groups),
            min_n_groups,
            [sum(nums[i] for i in group) for group in groups],
            max(group_percent),
            min(group_percent),
            np.mean(group_percent),
            time.monotonic() - st,
        )
