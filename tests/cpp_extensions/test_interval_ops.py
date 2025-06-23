# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import time
from typing import *

import numpy as np
import pytest
import torch

from realhf.impl.model.nn.flatten_param import (
    _set_intervals_py,
    _slice_intervals_py,
    set_intervals,
    slice_intervals,
)


def make_intervals(maxsize, n_intervals):
    assert maxsize // n_intervals > 1
    s = maxsize // n_intervals
    intervals = []
    interval_size = 0
    max_interval_size = 0
    for i in range(n_intervals):
        intervals.append((i * s, i * s + s // 2))
        interval_size += s // 2
        max_interval_size = max(max_interval_size, s // 2)
    np.random.shuffle(intervals)
    return np.array(intervals, dtype=np.int64), interval_size, max_interval_size


def maybe_synchronize_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@pytest.mark.skip("interval_ops are not used now.")
@pytest.mark.parametrize(
    "n_intervals", list(reversed([1, 100, 500, 1000, 2000, 4000, 10000, 100000]))
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float16])
@pytest.mark.gpu
def test_get(n_intervals: int, dtype: torch.dtype):
    device = torch.device("cuda")

    input_tensor = torch.randn(int(1e8), device=device, dtype=dtype)
    intervals, output_size, max_interval_size = make_intervals(
        input_tensor.size(0), n_intervals
    )
    intervals_cuda = torch.tensor(intervals, dtype=torch.long, device="cuda")

    # warmup
    slice_intervals(
        input_tensor,
        intervals,
        intervals_cuda=intervals_cuda,
        output_size=output_size,
        max_interval_size=max_interval_size,
    )
    _slice_intervals_py(input_tensor, intervals)

    maybe_synchronize_cuda()
    tik = time.perf_counter()
    for _ in range(10):
        output_tensor = slice_intervals(
            input_tensor,
            intervals,
            intervals_cuda=intervals_cuda,
            output_size=output_size,
            max_interval_size=max_interval_size,
        )
    maybe_synchronize_cuda()
    t1 = time.perf_counter() - tik

    maybe_synchronize_cuda()
    tik = time.perf_counter()
    for _ in range(10):
        o2 = _slice_intervals_py(input_tensor, intervals)
    maybe_synchronize_cuda()
    t2 = time.perf_counter() - tik
    assert torch.allclose(output_tensor, o2)
    print(
        f"slice_interval, Success! #intervals: {n_intervals} C++ ext time: {t1:.4f}, PyTorch time: {t2:.4f}"
    )


@pytest.mark.skip("interval_ops are not used now.")
@pytest.mark.parametrize(
    "n_intervals", list(reversed([1, 10, 100, 500, 1000, 1000, 10000, 100000]))
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.gpu
def test_set(n_intervals: int, dtype: torch.dtype):
    # NOTE: Since the set_intervals degenerate to the python implementation with CPU tensors,
    # We don't need to test it with CPU tensors.

    x = torch.randn(int(1e8), device="cuda", dtype=dtype)
    intervals, interval_size, max_interval_size = make_intervals(x.size(0), n_intervals)
    intervals_cuda = torch.tensor(intervals, dtype=torch.long, device="cuda")
    src = torch.randn(interval_size, device="cuda", dtype=dtype)

    # warmup
    input_tensor1 = x.clone()
    set_intervals(
        src,
        input_tensor1,
        intervals,
        intervals_cuda=intervals_cuda,
        max_interval_size=max_interval_size,
    )
    input_tensor2 = x.clone()
    _set_intervals_py(src, input_tensor2, intervals)

    input_tensor1 = x.clone()
    maybe_synchronize_cuda()
    tik = time.perf_counter()
    for _ in range(10):
        set_intervals(
            src,
            input_tensor1,
            intervals,
            intervals_cuda=intervals_cuda,
            max_interval_size=max_interval_size,
        )
    maybe_synchronize_cuda()
    t1 = time.perf_counter() - tik

    input_tensor2 = x.clone()
    maybe_synchronize_cuda()
    tik = time.perf_counter()
    for _ in range(10):
        _set_intervals_py(src, input_tensor2, intervals)
    maybe_synchronize_cuda()
    t2 = time.perf_counter() - tik

    assert torch.allclose(input_tensor1, input_tensor2)
    print(
        f"set_interval, Success! #intervals: {n_intervals}, C++ ext time: {t1:.4f}, PyTorch time: {t2:.4f}"
    )
