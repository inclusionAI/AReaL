# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import time

import pytest
import torch

from realhf.impl.model.utils.ppo_functional import (
    cugae1d_nolp_misalign_func,
    pygae1d_nolp_misalign,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="This test requires a GPU.")
@pytest.mark.parametrize("max_seqlen", [32, 128, 512])
@pytest.mark.parametrize("bs", [2, 4])
@pytest.mark.parametrize("gamma", [0.9, 1.0])
@pytest.mark.parametrize("lam", [0.5, 1.0])
@pytest.mark.gpu
def test_gae1d_nolp_misalign(max_seqlen: int, bs: int, gamma: float, lam: float):

    seqlens = torch.randint(1, max_seqlen, (bs,), dtype=torch.int32, device="cuda")
    rewards = torch.randn(seqlens.sum(), dtype=torch.float32, device="cuda")
    values = torch.randn(seqlens.sum() + bs, dtype=torch.float32, device="cuda")
    bootstrap = torch.ones(bs, dtype=torch.bool, device="cuda")
    cu_seqlens = torch.nn.functional.pad(seqlens.cumsum(0), (1, 0)).int()

    adv, ret = cugae1d_nolp_misalign_func(
        rewards, values, cu_seqlens, bootstrap, gamma, lam
    )
    py_adv, py_ret = pygae1d_nolp_misalign(
        rewards, values, cu_seqlens, bootstrap, gamma, lam
    )

    torch.cuda.synchronize()
    t1 = time.perf_counter_ns()
    py_adv, py_ret = pygae1d_nolp_misalign(
        rewards, values, cu_seqlens, bootstrap, gamma, lam
    )
    torch.cuda.synchronize()
    t2 = time.perf_counter_ns()
    adv, ret = cugae1d_nolp_misalign_func(
        rewards, values, cu_seqlens, bootstrap, gamma, lam
    )
    torch.cuda.synchronize()
    t3 = time.perf_counter_ns()

    assert torch.allclose(adv, py_adv, atol=1e-5), (adv - py_adv).abs().max()
    assert torch.allclose(ret, py_ret, atol=1e-5), (ret - py_ret).abs().max()

    print(
        f"max_seqlen={max_seqlen},bs={bs}, CUDA acceleration ratio",
        (t2 - t1) / (t3 - t2),
    )
