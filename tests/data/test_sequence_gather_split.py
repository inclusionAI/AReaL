# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import copy
import dataclasses
import itertools
import random
import uuid
from typing import *

import numpy as np
import pytest
import torch

from realhf.api.core.data_api import MicroBatchSpec, SequenceSample, SequenceSplitSpec
from realhf.base import datapack


def flatten_list(l: List[List]):
    return list(itertools.chain(*l))


def _make_sample_single_sequence(bs, data_with_none=False):
    keys = [
        "input_ids",
        "rewards",
        "logprobs",
        "logits_mask",
        "prompt_mask",
    ]
    vocab_size = 150
    slens = [int(torch.randint(1, 100, (1,)).int()) for _ in range(bs)]
    input_ids = torch.cat([torch.randint(0, 150, (slen,)) for slen in slens])
    rewards = torch.cat([torch.randn(1) for _ in range(bs)])
    logprobs = torch.cat([torch.randn(slen - 1) for slen in slens])
    if not data_with_none:
        logits_mask = torch.cat(
            [
                torch.randint(0, 2, (slen, vocab_size), dtype=torch.bool)
                for slen in slens
            ]
        )
    else:
        logits_mask = None
    prompt_mask = torch.cat(
        [torch.randint(0, 2, (slen,), dtype=torch.bool) for slen in slens]
    )
    data = dict(
        input_ids=input_ids,
        rewards=rewards,
        logprobs=logprobs,
        logits_mask=logits_mask,
        prompt_mask=prompt_mask,
    )
    ids = [uuid.uuid4() for _ in range(bs)]
    return SequenceSample.from_default(
        ids=ids,
        seqlens=slens,
        data=data,
        metadata=dict(a=[1 for _ in range(bs)], b=["abc" for _ in range(bs)]),
    )


def _make_sample_pair_sequence(bs):
    keys = [
        "group_factor",
        "seqlogprobs",
        "input_ids",
    ]
    slens = [torch.randint(1, 100, (2,)).int() for _ in range(bs)]
    input_ids = torch.cat(
        flatten_list([[torch.randint(0, 150, (s,)) for s in slen] for slen in slens])
    )
    group_factor = torch.cat([torch.randn(1) for _ in range(bs)])
    seqlogprobs = torch.cat([torch.randn(2, dtype=torch.float16) for _ in range(bs)])

    data = dict(
        input_ids=input_ids,
        group_factor=group_factor,
        seqlogprobs=seqlogprobs,
    )
    ids = [uuid.uuid4() for _ in range(bs)]
    trailing_shapes = dict(
        input_ids=(),
        group_factor=(),
        seqlogprobs=(),
    )
    dtypes = dict(
        input_ids=torch.long,
        group_factor=torch.float,
        seqlogprobs=torch.float16,
    )
    seqlens = dict(
        input_ids=slens,
        group_factor=[[1] for _ in range(bs)],
        seqlogprobs=[[1, 1] for _ in range(bs)],
    )
    return SequenceSample(
        keys=keys,
        ids=ids,
        seqlens=seqlens,
        trailing_shapes=trailing_shapes,
        dtypes=dtypes,
        data=data,
        metadata=dict(a=[1 for _ in range(bs)], b=["abc" for _ in range(bs)]),
    )


def _make_sample_group_sequence(bs):
    keys = [
        "input_ids",
        "logprobs",
    ]
    slens = [torch.randint(1, 100, (8,)).int() for _ in range(bs)]
    input_ids = torch.cat(
        flatten_list([[torch.randint(0, 150, (s,)) for s in slen] for slen in slens])
    )
    logprobs = torch.cat(
        flatten_list(
            [
                [torch.randn((s - 1,), dtype=torch.float16) for s in slen]
                for slen in slens
            ]
        )
    )

    data = dict(
        input_ids=input_ids,
        logprobs=logprobs,
    )
    ids = [uuid.uuid4() for _ in range(bs)]
    trailing_shapes = dict(
        input_ids=(),
        logprobs=(),
    )
    dtypes = dict(
        input_ids=torch.long,
        logprobs=torch.float16,
    )
    seqlens = dict(
        input_ids=slens,
        logprobs=[[s - 1 for s in slen] for slen in slens],
    )
    return SequenceSample(
        keys=keys,
        ids=ids,
        seqlens=seqlens,
        trailing_shapes=trailing_shapes,
        dtypes=dtypes,
        data=data,
    )


def _make_sample_single_prompt_multi_response(bs):
    keys = [
        "input_ids",
        "seq",
        "prompt_mask",
    ]
    n_ans_per_prompt = 5

    prompt_lens = [torch.randint(1, 100, (1,)).int() for _ in range(bs)]
    gen_lens = [torch.randint(1, 100, (n_ans_per_prompt,)).int() for _ in range(bs)]

    input_ids = torch.cat([torch.randint(0, 150, (slen,)) for slen in prompt_lens])
    seq = torch.cat(
        flatten_list([[torch.randint(0, 150, (s,)) for s in slen] for slen in gen_lens])
    )
    prompt_mask = torch.randint_like(seq, 0, 2, dtype=torch.bool)

    data = dict(
        input_ids=input_ids,
        seq=seq,
        prompt_mask=prompt_mask,
    )
    ids = [uuid.uuid4() for _ in range(bs)]
    trailing_shapes = dict(
        input_ids=(),
        seq=(),
        prompt_mask=(),
    )
    dtypes = dict(
        input_ids=torch.long,
        seq=torch.long,
        prompt_mask=torch.bool,
    )
    seqlens = dict(
        input_ids=[x.numpy().tolist() for x in prompt_lens],
        seq=[x.numpy().tolist() for x in gen_lens],
        prompt_mask=[x.numpy().tolist() for x in gen_lens],
    )
    return SequenceSample(
        keys=keys,
        ids=ids,
        seqlens=seqlens,
        trailing_shapes=trailing_shapes,
        dtypes=dtypes,
        data=data,
    )


def recursive_assert_equal(x1, x2):
    if type(x1) != type(x2):
        raise AssertionError(f"{type(x1)} != {type(x2)}")
    if isinstance(x1, dict):
        assert set(x1.keys()) == set(x2.keys())
        for k in x1.keys():
            recursive_assert_equal(x1[k], x2[k])
    elif dataclasses.is_dataclass(x1):
        for f in dataclasses.fields(x1):
            recursive_assert_equal(getattr(x1, f.name), getattr(x2, f.name))
    elif isinstance(x1, torch.Tensor):
        assert torch.allclose(x1, x2)
    elif isinstance(x1, list):
        assert len(x1) == len(x2)
        for a, b in zip(x1, x2):
            recursive_assert_equal(a, b)
    else:
        assert x1 == x2


@pytest.mark.parametrize(
    "sample_type", ["single", "single_with_none", "pair", "multi_sample", "group"]
)
@pytest.mark.parametrize("dp", [1, 2, 3, 4, 8, 15, 16])
def test_gather_split(sample_type: str, dp: int):
    batch_sizes = [random.randint(10, 20) for _ in range(dp)]
    if sample_type == "single":
        samples = [_make_sample_single_sequence(bs) for bs in batch_sizes]
    elif sample_type == "single_with_none":
        samples = [
            _make_sample_single_sequence(bs, data_with_none=True) for bs in batch_sizes
        ]
    elif sample_type == "pair":
        samples = [_make_sample_pair_sequence(bs) for bs in batch_sizes]
    elif sample_type == "group":
        samples = [_make_sample_group_sequence(bs) for bs in batch_sizes]
    elif sample_type == "multi_sample":
        samples = [_make_sample_single_prompt_multi_response(bs) for bs in batch_sizes]
    else:
        raise NotImplementedError()

    x = SequenceSample.gather(samples)

    # Test split to original samples
    spec = SequenceSplitSpec(sizes=batch_sizes)
    ss = x.split_with_spec(spec)
    for s1, s2 in zip(samples, ss):
        recursive_assert_equal(s1, s2)

    # Test json serialize
    import orjson

    bytes = orjson.dumps(x.as_json_compatible())
    y = SequenceSample.from_json_compatible(orjson.loads(bytes))
    recursive_assert_equal(x, y)

    # Test split to the finest granularity
    total_bs = sum(batch_sizes)
    ss, _, backward_indices = x.split(MicroBatchSpec(n_mbs=x.bs))
    assert len(ss) == total_bs
    y = SequenceSample.reorder(SequenceSample.gather(ss), backward_indices)
    recursive_assert_equal(x, y)

    # Test divide micro batch and merge back
    for seqlens in [
        [[s - 1 for s in slen] for slen in x.seqlens["input_ids"]],
        [[1 for _ in slen] for slen in x.seqlens["input_ids"]],
        [[s for s in slen] for slen in x.seqlens["input_ids"]],
    ]:
        output = [[torch.randn(s) for s in slen] for slen in seqlens]
        mb_spec = MicroBatchSpec(
            n_mbs=np.random.randint(1, 10),
            max_tokens_per_mb=np.random.randint(800, 1000),
        )
        mb_data, fwd_indices, bwd_indices = x.split(mb_spec)

        for id_x, id_y in zip(
            [x.ids[i] for i in fwd_indices],
            itertools.chain.from_iterable([xx.ids for xx in mb_data]),
        ):
            assert id_x == id_y

        xx = SequenceSample.reorder(SequenceSample.gather(mb_data), bwd_indices)
        recursive_assert_equal(x, xx)

        actual_output = torch.cat([torch.cat(output[i]) for i in fwd_indices], dim=0)
        reordered = SequenceSample.reorder_output(
            actual_output,
            expected_seqlens=seqlens,
            forward_indices=fwd_indices,
            backward_indices=bwd_indices,
        )
        assert torch.allclose(torch.cat(datapack.flat2d(output)), reordered)
