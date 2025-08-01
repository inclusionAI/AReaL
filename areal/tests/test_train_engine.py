# Copyright 2025 Ant Group Inc.
# Licensed under the Apache License, Version 2.0

"""Test script for Engine implementation."""

import os
from typing import Dict

import pytest
import torch
from tensordict import TensorDict
from transformers import AutoTokenizer

from areal.api.cli_args import MicroBatchSpec, OptimizerConfig, TrainEngineConfig
from areal.api.io_struct import FinetuneSpec, SaveLoadMeta

VOCAB_SIZE = 100
MODEL_PATH = "/storage/testing/models/Qwen__Qwen3-1.7B/"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "Qwen/Qwen2-0.5B"


@pytest.fixture(scope="module")
def mock_input(
    batch_size=5,
    min_seqlen=10,
    max_seqlen=20,
    device="cuda:0",
) -> Dict:
    """Create mock padded input data (same format for huggingface) for testing.
    Returns a dict with input_ids, attention_mask, and position_ids.
    """
    pad_token_id = 0
    seqlens = torch.randint(
        min_seqlen, max_seqlen, (batch_size,), dtype=torch.int, device=device
    )
    max_seqlen = int(max(seqlens))
    input_ids = torch.randint(
        0, VOCAB_SIZE, (batch_size, max_seqlen), dtype=torch.long, device=device
    )
    attn_mask = torch.zeros((batch_size, max_seqlen), dtype=torch.bool, device=device)

    attn_mask[
        torch.arange(0, max_seqlen, device=device).unsqueeze(0) < seqlens.unsqueeze(1)
    ] = 1
    input_ids.masked_fill_(~attn_mask, pad_token_id)

    return TensorDict(
        input_ids=input_ids,
        attention_mask=attn_mask,
    )


def get_engine(engine_type: str, model_path: str):
    from areal.engine.fsdp_engine import FSDPEngine
    from areal.experimental.autotp_engine import DeepSpeedAutoTPEngine

    engine_cls = {"auto_tp": DeepSpeedAutoTPEngine, "fsdp": FSDPEngine}[engine_type]

    engine_config = TrainEngineConfig(
        experiment_name=f"test-{engine_type}-engine",
        trial_name="test0",
        path=model_path,
        optimizer=OptimizerConfig(),
    )
    engine = engine_cls(engine_config)
    ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=100, train_batch_size=2)
    engine.initialize(None, ft_spec)
    return engine


def mock_loss_fn(logits: torch.Tensor, input_data: Dict) -> torch.Tensor:
    """Mock loss function for testing."""
    return torch.mean(logits)


@pytest.fixture(scope="module", params=["fsdp", "auto_tp"])
def engine(request):
    os.environ.update(
        {
            "WORLD_SIZE": "1",
            "RANK": "0",
            "LOCAL_RANK": "0",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "7777",
        }
    )

    engine = get_engine(request.param, MODEL_PATH)
    print(f"✓ {request.param.upper()} Engine created successfully")
    yield engine


@torch.no_grad()
def test_forward_microbatch(engine, mock_input):
    engine.eval()
    engine.config.mb_spec = MicroBatchSpec(n_mbs=2, max_tokens_per_mb=100)
    x2 = engine.forward(input_=mock_input).squeeze(0).mean(-1)
    engine.config.mb_spec = MicroBatchSpec(n_mbs=1, max_tokens_per_mb=100)
    x1 = engine.forward(input_=mock_input).squeeze(0).mean(-1)
    input_ids = mock_input["input_ids"]
    assert x1.shape[:1] == input_ids.shape[:1]
    assert x2.shape[:1] == input_ids.shape[:1]
    assert torch.allclose(x1, x2, atol=1e-1, rtol=1e-2), (x1 - x2).abs().max().item()


@torch.no_grad()
def test_eval_batch(engine, mock_input):
    engine.eval()
    engine.config.mb_spec = MicroBatchSpec(n_mbs=2, max_tokens_per_mb=100)
    eval_result = engine.eval_batch(
        input_=mock_input,
        loss_fn=mock_loss_fn,
        loss_weight_fn=lambda x: x["cu_seqlens"][-1],
    )
    assert isinstance(eval_result, torch.Tensor), "Evaluation should return a tensor"
    assert eval_result.is_cuda, "Evaluation tensor should be on CUDA device"
    assert eval_result is not None, "Evaluation should return a loss value"
    print(f"✓ Evaluation successful, loss: {eval_result.item()}")


def test_train_batch(engine, mock_input):
    engine.train()
    engine.config.mb_spec = MicroBatchSpec(n_mbs=2, max_tokens_per_mb=100)
    train_result = engine.train_batch(
        input_=mock_input,
        loss_fn=mock_loss_fn,
        loss_weight_fn=lambda x: x["cu_seqlens"][-1],
    )
    assert isinstance(train_result, dict), "Training should return a dictionary"
    assert train_result["grad_norm"] is not None
    assert train_result["lr"] is not None
    print("✓ Training successful")


@torch.no_grad()
def test_hf_save_load_weights(tmp_path_factory, engine, mock_input):
    from areal.experimental.autotp_engine import DeepSpeedAutoTPEngine

    if isinstance(engine, DeepSpeedAutoTPEngine):
        print("AutoTP engine does not support HF save/load for now.")
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    path = tmp_path_factory.mktemp("hf_engine_test")
    save_load_meta = SaveLoadMeta(
        path=path,
        weight_format="hf",
        tokenizer=tokenizer,
        with_optim=True,
        base_model_path=None,
    )

    engine.config.mb_spec = MicroBatchSpec(n_mbs=1, max_tokens_per_mb=100)
    old = engine.forward(input_=mock_input)
    engine.save(save_load_meta)

    for name, param in engine.model.named_parameters():
        param.zero_()

    engine.load(save_load_meta)
    new = engine.forward(input_=mock_input)
    assert torch.allclose(old, new)
