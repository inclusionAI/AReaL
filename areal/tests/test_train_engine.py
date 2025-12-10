"""Test script for Engine implementation."""

import os
from typing import Any

import pytest
import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from areal.api.cli_args import MicroBatchSpec, OptimizerConfig, TrainEngineConfig
from areal.api.io_struct import FinetuneSpec, SaveLoadMeta
from areal.platforms import current_platform
from areal.tests.utils import get_model_path

VOCAB_SIZE = 100
MODEL_PATH = get_model_path(
    "/storage/openpsi/models/Qwen__Qwen3-0.6B/", "Qwen/Qwen3-0.6B"
)


@pytest.fixture(scope="module")
def mock_input(
    batch_size=5,
    min_seqlen=10,
    max_seqlen=20,
    device=current_platform.device_type,
) -> dict[str, Any]:
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

    return dict(
        input_ids=input_ids,
        attention_mask=attn_mask,
    )


def get_engine(engine_type: str, model_path: str):
    from areal.engine.fsdp_engine import FSDPEngine

    engine_cls = {"fsdp": FSDPEngine}[engine_type]

    engine_config = TrainEngineConfig(
        experiment_name=f"test-{engine_type}-engine",
        trial_name="test0",
        path=model_path,
        optimizer=OptimizerConfig(),
    )
    engine = engine_cls(engine_config)
    engine.create_process_group()
    ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=100, train_batch_size=2)
    engine.initialize(None, ft_spec)
    return engine


def mock_loss_fn(
    logprobs: torch.Tensor,
    entropy: torch.Tensor,
    input_data: dict,
    **kwargs,
) -> torch.Tensor:
    """Mock loss function for testing."""
    return torch.mean(logprobs)


@pytest.fixture(params=["fsdp"])
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
    try:
        yield engine
    finally:
        engine.destroy()
        assert not dist.is_initialized()


@torch.no_grad()
def test_forward_microbatch(engine, mock_input):
    engine.eval()
    engine.config.mb_spec = MicroBatchSpec(n_mbs=2, max_tokens_per_mb=100)
    x2 = engine.forward(input_=mock_input)
    engine.config.mb_spec = MicroBatchSpec(n_mbs=1, max_tokens_per_mb=100)
    x1 = engine.forward(input_=mock_input)

    attn_mask = mock_input["attention_mask"]
    loss_mask = attn_mask.clone()
    loss_mask[:, :-1] = attn_mask[:, :-1] & attn_mask[:, 1:]
    loss_mask[:, -1] = False

    x1_valid = x1[loss_mask]
    x2_valid = x2[loss_mask]
    assert torch.allclose(x1_valid, x2_valid, atol=1e-4, rtol=1e-3), (
        (x1_valid - x2_valid).abs().max().item()
    )


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


@torch.no_grad()
def test_dcp_save_load_weights(tmp_path_factory, engine, mock_input):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    path = tmp_path_factory.mktemp("dcp_engine_test")
    save_load_meta = SaveLoadMeta(
        path=path,
        weight_format="dcp",
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
