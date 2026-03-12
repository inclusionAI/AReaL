import bisect
import copy
import dataclasses
import itertools
from typing import Any

import numba
import numpy as np
import torch


def flat2d(arr: list[list[Any]]) -> list[Any]:
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
) -> list[tuple[int, int]]:
    lst = partition_balanced(nums, k, min_size)
    return [(lst[i], lst[i + 1]) for i in range(k)]


def min_abs_diff_partition(
    arr: np.ndarray | list, k: int, min_size: int = 1
) -> list[tuple[int, int]]:
    err_hint = (
        " Errors should not be reported in this function. It is probably a bug in the dataset code"
        " or too small batch size with pipeline parallelism."
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
) -> tuple[np.ndarray, int]:
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
    values: np.ndarray, capacity: int, min_groups: int, n_groups_divisor: int = 1
) -> list[list[int]]:
    """A greedy allocation algorithm that partitions a list of numbers
    into k groups, where the summation of each group is less than capacity
    and (k >= min_groups and k % n_groups_divisor == 0). We want to minimize
    k and make partitions as balanced as possible.

    1. Sort the numbers in reverse order.
    2. If the number of groups is less than `min_groups`, create a new group.
    3. For a new number, find all groups with the capacity to hold the new number.
       Put the new number into the group with the smallest size.
    4. Otherwise, create a new group.
    """
    value_indices = np.argsort(-values)
    group_indices: list[list[int]] = []
    group_values: list[tuple[float, int]] = []
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


def ffd_allocate(
    values: list[int], capacity: int, min_groups: int, n_groups_divisor: int = 1
) -> list[list[int]]:
    if min_groups is None or min_groups < n_groups_divisor:
        min_groups = n_groups_divisor
    if any(v > capacity for v in values):
        raise RuntimeError(f"Values {values} is larger than capacity {capacity}")
    if len(values) < min_groups:
        raise RuntimeError(
            f"Number of values {len(values)} is smaller than min_groups {min_groups}"
        )
    while True:
        res = _ffd_allocate(np.array(values), capacity, min_groups)
        min_groups += n_groups_divisor - min_groups % n_groups_divisor
        if len(res) % n_groups_divisor == 0:
            break
        if len(values) < min_groups:
            raise RuntimeError(
                f"Cannot allocate values {values} that satisfies capacity {capacity}, "
                f"min_groups {min_groups} and n_groups_divisor {n_groups_divisor}."
            )
    return res


def balanced_greedy_partition(nums: list[int], K: int) -> list[list[int]]:
    """
    Splits `nums` into K groups such that the maximum difference between group sums is minimized.

    Returns indices (not values) for each group.

    Greedy with capacity-aware assignment.

    Args:
        nums: List of values to partition
        K: Number of groups to partition into

    Returns:
        List of K lists, where each inner list contains the indices assigned to that group

    Raises:
        ValueError: If len(nums) is not divisible by K or if len(nums) < K
    """
    n = len(nums)
    if n < K:
        raise ValueError(f"Number of items ({n}) must be >= K ({K}).")
    if n % K != 0:
        raise ValueError("The length of nums must be divisible by K.")
    m = n // K

    # Sort indices by value in descending order
    sorted_indices = sorted(range(n), key=lambda i: -nums[i])

    groups: list[list[int]] = [[] for _ in range(K)]
    sums = [0 for _ in range(K)]
    counts = [0 for _ in range(K)]

    for idx in sorted_indices:
        num = nums[idx]
        # Find the non-full group with the smallest current sum
        chosen_group = -1
        min_sum = float("inf")
        for i in range(K):
            if counts[i] < m and sums[i] < min_sum:
                min_sum = sums[i]
                chosen_group = i

        groups[chosen_group].append(idx)
        sums[chosen_group] += num
        counts[chosen_group] += 1

    return groups


def _pad_cat_dim0(tensors: list[torch.Tensor]) -> torch.Tensor:
    """Pad tensors to same non-batch dims and concatenate along dim 0."""
    # Get the maximum shape for dims 1 to N-1
    shape = [0 for _ in range(tensors[0].ndim - 1)]
    for t in tensors:
        if t.ndim != len(shape) + 1:
            raise ValueError(
                f"Shard dimension mismatch: expected {len(shape) + 1}, got {t.ndim}"
            )
        for i in range(1, t.ndim):
            shape[i - 1] = max(shape[i - 1], t.shape[i])

    # Pad tensors
    padded_tensors = []
    for t in tensors:
        pad_sizes = []
        for i in range(1, t.ndim):
            pad_size = shape[i - 1] - t.shape[i]
            pad_sizes.append(pad_size)
        if any(pad_sizes):
            pad = []
            for pad_size in reversed(pad_sizes):
                pad.extend([0, pad_size])
            pt = torch.nn.functional.pad(t, tuple(pad), "constant", 0)
            padded_tensors.append(pt)
            continue
        padded_tensors.append(t)

    return torch.cat(padded_tensors, dim=0)


def pad_and_concat_tensors(dicts: list[dict[str, Any]]) -> dict[str, Any]:
    """Concat a list of trajectory dicts (with plain tensors) into a single batch dict."""
    if not dicts:
        return {}
    if len(dicts) == 1:
        return dicts[0]
    # Validate key consistency
    first_keys = set(dicts[0].keys())
    for i, d in enumerate(dicts[1:], 1):
        if set(d.keys()) != first_keys:
            raise ValueError(
                f"pad_and_concat_tensors: dict[{i}] has different keys than dict[0]. "
                f"Expected {sorted(first_keys)}, got {sorted(d.keys())}"
            )
    result: dict[str, Any] = {}
    for key in dicts[0]:
        values = [d[key] for d in dicts]
        if isinstance(values[0], torch.Tensor):
            result[key] = _pad_cat_dim0(values)
        elif isinstance(values[0], list):
            result[key] = [item for v in values for item in v]
        else:
            result[key] = values[0]
    return result


def _unpad_splits(
    splits: list[torch.Tensor], traj_seqlens: list[int] | None
) -> list[torch.Tensor]:
    """Trim each split tensor's last dim to its original sequence length."""
    if traj_seqlens is None:
        return splits
    for i, s in enumerate(splits):
        if s.ndim >= 2 and s.shape[-1] > traj_seqlens[i]:
            splits[i] = s[..., : traj_seqlens[i]]
    return splits


def split_and_unpad_tensor(
    result: Any,
    n_trajs: int,
    traj_group_sizes: list[int] | int = 1,
    traj_seqlens: list[int] | None = None,
) -> Any:
    """Split a batched result back into per-trajectory list, optionally unpadding.

    Inverse of pad_and_concat_tensors for engine outputs. Handles:
    - torch.Tensor → list[torch.Tensor] (split along dim 0)
    - dict[str, Tensor] → list[dict[str, Tensor]]
    - None → None

    When traj_seqlens is provided, each split tensor is trimmed along the last
    dimension to its original sequence length (undoing the padding from
    pad_and_concat_tensors).

    Parameters
    ----------
    result : Any
        Batched result from engine computation.
    n_trajs : int
        Number of trajectories to split into.
    traj_group_sizes : list[int] | int
        Per-trajectory batch sizes (dim 0) for splitting. Accepts a single int
        for uniform sizes (backward compatibility). Default 1.
    traj_seqlens : list[int] | None
        Per-trajectory sequence lengths for unpadding. Default None (no unpadding).

    Returns
    -------
    Any
        Per-trajectory results as a list, or None.
    """
    if result is None:
        return None
    # Normalize to list for uniform handling
    if isinstance(traj_group_sizes, int):
        traj_group_sizes = [traj_group_sizes] * n_trajs
    total = sum(traj_group_sizes)
    if isinstance(result, torch.Tensor):
        splits = list(result.split(traj_group_sizes, dim=0))
        return _unpad_splits(splits, traj_seqlens)
    if isinstance(result, dict):
        split_result = [{} for _ in range(n_trajs)]
        for key, value in result.items():
            if isinstance(value, torch.Tensor) and value.shape[0] == total:
                splits = _unpad_splits(
                    list(value.split(traj_group_sizes, dim=0)), traj_seqlens
                )
                for i, s in enumerate(splits):
                    split_result[i][key] = s
            else:
                for i in range(n_trajs):
                    split_result[i][key] = copy.deepcopy(value)
        return split_result
    return result


@dataclasses.dataclass
class TrajBatchMeta:
    """Metadata captured by pack_batch for unpack_batch to reverse the transformation.

    Carries context needed to symmetrically undo the concat+pad done by
    pack_batch, including per-trajectory batch sizes and sequence lengths
    for unpadding engine outputs that may have been padded to a different max.
    """

    n_trajs: int
    traj_group_sizes: list[int]
    traj_seqlens: list[int]


def pack_batch(
    data: list[dict[str, Any]],
) -> tuple[dict[str, Any], TrajBatchMeta]:
    """Concat list[dict] trajectories into a single batched dict.

    Parameters
    ----------
    data : list[dict[str, Any]]
        List of trajectory dicts to concatenate.

    Returns
    -------
    tuple[dict[str, Any], TrajBatchMeta]
        (batched_dict, meta) where meta carries n_trajs, per-traj batch sizes,
        and per-traj sequence lengths for later unpadding.
    """
    assert isinstance(data, list) and all(isinstance(d, dict) for d in data), (
        f"Expected list[dict], got {type(data)}"
    )
    traj_group_sizes = []
    for d in data:
        first_tensor = next(
            (v for v in d.values() if isinstance(v, torch.Tensor)), None
        )
        traj_group_sizes.append(
            first_tensor.shape[0] if first_tensor is not None else 1
        )
    traj_seqlens = [d["attention_mask"].shape[-1] for d in data]
    meta = TrajBatchMeta(
        n_trajs=len(data),
        traj_group_sizes=traj_group_sizes,
        traj_seqlens=traj_seqlens,
    )
    return pad_and_concat_tensors(data), meta


def unpack_batch(
    result: Any,
    meta: TrajBatchMeta,
) -> list[Any] | None:
    """Split batched result back into per-trajectory list and unpad.

    Parameters
    ----------
    result : Any
        Batched result from engine computation.
    meta : TrajBatchMeta
        Metadata from pack_batch containing per-traj sequence lengths.

    Returns
    -------
    list[Any] | None
        Per-trajectory results as a list, or None.
    """
    return split_and_unpad_tensor(
        result, meta.n_trajs, meta.traj_group_sizes, meta.traj_seqlens
    )


def find_in_structure(obj: Any, type_: type) -> Any | None:
    """Find first instance of type_ in a nested structure."""
    if isinstance(obj, type_):
        return obj
    if isinstance(obj, dict):
        for v in obj.values():
            result = find_in_structure(v, type_)
            if result is not None:
                return result
    if isinstance(obj, (tuple, list)):
        for item in obj:
            result = find_in_structure(item, type_)
            if result is not None:
                return result
    return None


def dispatch_traj_list(
    traj_list: list[dict[str, Any]],
    dp_size: int,
) -> tuple[list[list[dict[str, Any]]], list[list[int]]]:
    """Partition a trajectory list across DP groups by balanced token count.

    Parameters
    ----------
    traj_list : list[dict[str, Any]]
        List of trajectory dicts. Dicts may contain RTensor values whose
        shard seqlens are used as partition weights.
    dp_size : int
        Number of data parallel groups.

    Returns
    -------
    tuple[list[list[dict[str, Any]]], list[list[int]]]
        (splits, group_indices) where splits[i] is the traj subset for DP group i.
    """
    from areal.infra.rpc.rtensor import RTensor

    seqlens = []
    for d in traj_list:
        for v in d.values():
            if isinstance(v, RTensor):
                seqlens.append(sum(v.shard.seqlens))
                break
        else:
            seqlens.append(1)

    group_indices = balanced_greedy_partition(seqlens, K=dp_size)
    splits = [[traj_list[i] for i in idxs] for idxs in group_indices]
    return splits, group_indices


def data_parallel_merge(results: list[Any]) -> Any:
    """Merge results from data parallel processing. Standalone version.

    Parameters
    ----------
    results : list[Any]
        Results from each DP group.

    Returns
    -------
    Any
        Merged result with original ordering restored.
    """
    from areal.infra.rpc.rtensor import RTensor  # Lazy import to avoid circular dep

    if not results:
        return None

    first = results[0]

    # Raw tensors and RTensors should never reach this merge path.
    # RTensors flow through _reorder_traj_results in the controller instead.
    if isinstance(first, torch.Tensor):
        raise TypeError(
            "Regular tensors not allowed in merge - only RTensors. "
            "Engine outputs should be automatically converted to RTensors."
        )

    if isinstance(first, RTensor):
        raise TypeError(
            "RTensors should not be merged via data_parallel_merge. "
            "Per-trajectory RTensors flow through _reorder_traj_results instead."
        )

    if isinstance(first, dict):
        merged = {}
        for key in first.keys():
            values = [r[key] for r in results]
            merged[key] = data_parallel_merge(values)
        return merged

    if isinstance(first, (list, tuple)):
        merged = [
            data_parallel_merge([r[i] for r in results]) for i in range(len(first))
        ]
        return type(first)(merged)

    # Scalars: return first (assume synchronized)
    return first


if __name__ == "__main__":
    import time

    for i in range(100):
        st = time.monotonic()
        nums = np.random.randint(1024, 8192, size=(100,)).tolist()
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
