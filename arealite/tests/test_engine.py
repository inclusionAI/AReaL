# Copyright 2025 Ant Group Inc.
# Licensed under the Apache License, Version 2.0

"""Test script for HF Engine implementation."""

import copy
from typing import Dict

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from arealite.api.cli_args import (
    EngineBackendConfig,
    EngineConfig,
    MicroBatchSpec,
    ModelFamily,
    OptimizerConfig,
    TrainingArgs,
)
from arealite.api.engine_api import EngineFactory
from arealite.api.io_struct import FinetuneSpec
from arealite.utils import compute_varlen_position_indices
from realhf.impl.model.utils.padding import unpad_input

VOCAB_SIZE = 100


@pytest.fixture(scope="module")
def mock_input(bs: int = 3, min_seqlen: int = 3, max_seqlen: int = 12) -> Dict:
    """Create mock input data for testing."""
    seqlens = torch.randint(
        min_seqlen, max_seqlen, (bs,), dtype=torch.int, device="cuda:0"
    )
    max_seqlen = int(max(seqlens))
    input_ids = torch.randint(
        0, VOCAB_SIZE, (bs, max_seqlen), dtype=torch.long, device="cuda:0"
    )

    attn_mask = torch.zeros((bs, max_seqlen), dtype=torch.bool, device="cuda:0")
    attn_mask[
        torch.arange(0, max_seqlen, device="cuda:0").unsqueeze(0) < seqlens.unsqueeze(1)
    ] = 1

    packed_input_ids, indices, cu_seqlens, max_seqlen = unpad_input(
        input_ids, attn_mask
    )

    assert torch.allclose(
        cu_seqlens, torch.nn.functional.pad(seqlens.cumsum(0, dtype=torch.int), (1, 0))
    )
    position_ids = compute_varlen_position_indices(int(sum(seqlens)), cu_seqlens)

    return dict(
        input_ids=packed_input_ids.unsqueeze(0),
        attention_mask=None,
        position_ids=position_ids.unsqueeze(0),
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        use_cache=False,
    )


def mock_loss_fn(logits: torch.Tensor, input_data: Dict) -> torch.Tensor:
    """Mock loss function for testing."""
    return torch.mean(logits)


@pytest.fixture(params=["hf", "fsdp"], scope="module")
def backend_type(request):
    return request.param


@pytest.fixture(scope="module")
def engine(backend_type):
    engine_config = EngineConfig(
        type=ModelFamily("qwen2", False),
        path="Qwen/Qwen2.5-0.5B",
        gradient_checkpointing=False,
        optimizer=OptimizerConfig(),
        backend=EngineBackendConfig(type=backend_type),
    )

    mock_args = TrainingArgs(n_nodes=1, n_gpus_per_node=1)

    engine_factory = EngineFactory(mock_args)
    engine = engine_factory.make_engine(engine_config)
    ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=100, train_batch_size=2)
    engine.init_distributed(None, ft_spec)
    print("✓ Engine created successfully")
    yield engine


def test_forward_microbatch(engine, mock_input):
    x2 = (
        engine.forward(
            input_=mock_input,
            mb_spec=MicroBatchSpec(n_mbs=2),
            aggregate_fn=lambda x: torch.cat(x, dim=1),
        )
        .squeeze(0)
        .mean(-1)
    )
    x1 = (
        engine.forward(
            input_=mock_input,
            mb_spec=MicroBatchSpec(n_mbs=1),
            aggregate_fn=lambda x: torch.cat(x, dim=1),
        )
        .squeeze(0)
        .mean(-1)
    )
    input_ids = mock_input["input_ids"].squeeze(0)
    assert x1.shape[0] == input_ids.shape[0]
    assert x2.shape[0] == input_ids.shape[0]
    assert torch.allclose(x1, x2, atol=1e-1, rtol=1e-2), (x1 - x2).abs().max().item()


def test_eval_batch(engine, mock_input):
    eval_result = engine.eval_batch(
        input_=mock_input,
        mb_spec=MicroBatchSpec(n_mbs=2),
        loss_fn=mock_loss_fn,
        loss_weight_fn=lambda x: x["cu_seqlens"][-1],
    )
    assert isinstance(eval_result, torch.Tensor), "Evaluation should return a tensor"
    assert eval_result.is_cuda, "Evaluation tensor should be on CUDA device"
    assert eval_result is not None, "Evaluation should return a loss value"
    print(f"✓ Evaluation successful, loss: {eval_result.item()}")


def test_train_batch(tmp_path_factory, engine, mock_input):
    path = tmp_path_factory.mktemp("hf_engine_train_batch")
    engine.save_optimizer_state(path)

    train_result = engine.train_batch(
        input_=mock_input,
        mb_spec=MicroBatchSpec(n_mbs=2),
        loss_fn=mock_loss_fn,
        loss_weight_fn=lambda x: x["cu_seqlens"][-1],
    )
    assert isinstance(train_result, dict), "Training should return a dictionary"
    assert train_result["grad_norm"] is not None
    assert train_result["lr"] is not None
    print("✓ Training successful")

    engine.load_optimizer_state(path)


def test_save_load_weights(tmp_path_factory, engine, mock_input):
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    path = tmp_path_factory.mktemp("hf_engine_test")

    engine.save_model_to_hf(path=path, tokenizer=tokenizer)
    old = engine.forward(
        input_=mock_input,
        mb_spec=MicroBatchSpec(n_mbs=1),
    )
    engine.load_model_from_hf(path=path)
    new = engine.forward(
        input_=mock_input,
        mb_spec=MicroBatchSpec(n_mbs=1),
    )

    assert torch.allclose(old, new)
