# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import random
import time
import uuid

import pytest
import torch

from realhf.base import constants, name_resolve, testing


# This is a test for grouped_gemm experts implementation of MoE.
@torch.no_grad()
def run_grouped_mlp(num_tokens, tp_size, token_dispatch_strategy, seed=1):
    # inline import to avoid torch re-initialize
    from realhf.api.core.model_api import ReaLModelConfig
    from realhf.impl.model.modules.moe.experts import GroupedMLP, SequentialMLP
    from realhf.impl.model.nn.real_llm_api import ReaLModel

    mconfig: ReaLModelConfig = getattr(ReaLModel, f"make_mixtral_config")()
    hidden_dim = mconfig.hidden_dim
    num_experts = mconfig.moe.num_experts
    assert num_tokens % num_experts == 0, f"{num_tokens} % {num_experts} != 0"
    torch.manual_seed(seed)
    random.seed(seed)

    testing.init_global_constants(
        num_dp=1,
        num_tp=tp_size,
        num_pp=1,
        sequence_parallel=False,  # grouped gemm does not support sequence parallel
        max_prompt_len=128,  # useless value in this test
    )
    device = torch.device("cuda")

    with constants.model_scope(testing.MODEL_NAME):
        seq_mlp = SequentialMLP(
            config=mconfig,
            dtype=torch.bfloat16,
            device=device,
        )
        seq_mlp_state_dict = seq_mlp.state_dict()

        grouped_mlp = GroupedMLP(
            config=mconfig,
            dtype=torch.bfloat16,
            device=device,
        )
        grouped_mlp.load_state_dict(seq_mlp_state_dict)

        permuted_hidden_states = torch.rand(
            (num_tokens, hidden_dim), dtype=torch.bfloat16, device=device
        )

        if token_dispatch_strategy == "even":
            tokens_per_expert = [num_tokens // num_experts for _ in range(num_experts)]
        elif token_dispatch_strategy == "random":
            tokens_left = num_tokens
            tokens_per_expert = []
            for _ in range(num_experts - 1):
                tokens_per_expert.append(random.randint(0, tokens_left))
                tokens_left -= tokens_per_expert[-1]
            tokens_per_expert.append(tokens_left)
        elif token_dispatch_strategy == "zero":
            tokens_per_expert = [0 for _ in range(num_experts - 1)] + [num_tokens]
        else:
            raise NotImplementedError()

        tokens_per_expert = torch.tensor(tokens_per_expert)

        o1 = seq_mlp(permuted_hidden_states, tokens_per_expert)
        o2 = grouped_mlp(permuted_hidden_states, tokens_per_expert)

        st = time.perf_counter()
        for _ in range(10):
            o1 = seq_mlp(permuted_hidden_states, tokens_per_expert)
        t1 = time.perf_counter() - st

        st = time.perf_counter()
        for _ in range(10):
            o2 = grouped_mlp(permuted_hidden_states, tokens_per_expert)
        t2 = time.perf_counter() - st

        print(
            f"rank {constants.tensor_parallel_rank()}: "
            f"{token_dispatch_strategy} diff: {(o1 - o2).abs().max()}: time {t1:.4f} {t2:.4f}"
        )
        # NOTE: With some input shapes, there are possibility that
        # GroupedMLP and SequentialMLP produce results of around 2% difference
        # due to grouped_gemm implementation
        assert torch.allclose(o1, o2, rtol=0.02), (
            constants.tensor_parallel_rank(),
            token_dispatch_strategy,
            (o1 - o2).abs().max(),
            o1.abs().max(),
        )


@pytest.mark.skip("grouped_gemm is not used for now.")
@pytest.mark.parametrize("num_tokens", [200])
@pytest.mark.parametrize("tp_size", [1, 2])
@pytest.mark.parametrize("token_dispatch_strategy", ["random"])
@pytest.mark.gpu
@pytest.mark.distributed
def test_grouped_mlp(
    num_tokens,
    tp_size,
    token_dispatch_strategy,
):
    test = testing.LocalMultiProcessTest(
        tp_size,
        run_grouped_mlp,
        num_tokens,
        tp_size,
        token_dispatch_strategy,
    )
    test.launch()


@pytest.mark.skip("grouped_gemm is not used for now.")
@pytest.mark.gpu
def test_grouped_gemm():
    torch.manual_seed(1)
    device = torch.device("cuda")
    import grouped_gemm as gg
    import torch.nn.functional as F

    w1 = torch.rand((128, 128), dtype=torch.bfloat16, device=device)
    w2 = torch.rand((4, 128, 160), dtype=torch.bfloat16, device=device)

    splits = torch.tensor([32, 32, 32, 32])
    o1 = gg.ops.gmm(w1, w2, splits, trans_b=False)

    o2 = torch.zeros((128, 160), dtype=torch.bfloat16, device=device)

    for i in range(4):
        wi = w2[i, :].squeeze_().transpose_(0, 1)
        o2[32 * i : 32 * (i + 1)] = F.linear(w1[32 * i : 32 * (i + 1)], wi)

    assert torch.allclose(o1, o2), (o1 - o2).abs().max()
