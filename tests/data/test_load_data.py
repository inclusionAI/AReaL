# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import random
import uuid

import pytest
import torch
from torch.utils.data import DataLoader

from realhf.api.core import config as config_api
from realhf.api.core import data_api
from tests.fixtures import *


def _validate_dataset(cfg: config_api.DatasetAbstraction, tokenizer):
    dataset = data_api.make_dataset(
        cfg,
        seed=1,
        dp_rank=0,
        world_size=1,
        tokenizer_or_tokenizer_name=tokenizer,
    )
    dataloader = DataLoader(
        dataset,
        collate_fn=data_api.SequenceSample.gather,
        # NOTE: This is *NOT* the actual batch size for training.
        # It is just a proper size to load data to workers.
        batch_size=10240,
        shuffle=True,
    )
    for x in dataloader:
        assert isinstance(x, data_api.SequenceSample)
        assert x.data is not None
        for k, v in x.data.items():
            assert v.device == torch.device("cpu")
        for k, vs in x.seqlens.items():
            assert all(isinstance(v, list) for v in vs)
            assert all(all(isinstance(vv, int) for vv in v) for v in vs)
        assert len(x.ids) == len(set(x.ids))
        if x.metadata:
            for k, v in x.metadata.items():
                assert isinstance(v, list), k
        xs = x.unpack()
        for xx in xs:
            if xx.metadata:
                for k, v in xx.metadata.items():
                    assert isinstance(v, list), k
                    assert len(v) == 1


@pytest.mark.parametrize("max_length", [16, 32, 128])
def test_prompt_answer_dataset(dataset, tokenizer, max_length: int):
    # NOTE: import all dataset implementations
    import realhf.impl.dataset

    cfg = config_api.DatasetAbstraction(
        type_="prompt_answer",
        args=dict(max_length=max_length, dataset_builder=lambda: dataset),
    )
    _validate_dataset(cfg, tokenizer)


@pytest.mark.parametrize("max_length", [16, 32, 128])
def test_prompt_only_dataset(
    dataset,
    tokenizer,
    max_length: int,
):
    # NOTE: import all dataset implementations
    import realhf.impl.dataset

    cfg = config_api.DatasetAbstraction(
        type_="prompt",
        args=dict(
            max_length=max_length,
            dataset_builder=lambda: dataset,
        ),
    )
    _validate_dataset(cfg, tokenizer)


@pytest.mark.parametrize("max_length", [16, 32, 128])
@pytest.mark.parametrize("max_pairs_per_prompt", [1, 3, 10])
def test_paired_rw_dataset(
    dataset, tokenizer, max_length: int, max_pairs_per_prompt: int
):
    # NOTE: import all dataset implementations
    import realhf.impl.dataset

    cfg = config_api.DatasetAbstraction(
        type_="rw_pair",
        args=dict(
            max_length=max_length,
            dataset_builder=lambda: dataset,
            max_pairs_per_prompt=max_pairs_per_prompt,
        ),
    )
    _validate_dataset(cfg, tokenizer)
