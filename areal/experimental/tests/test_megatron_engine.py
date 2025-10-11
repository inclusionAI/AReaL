import os
import time
from importlib.metadata import version as get_version

import pytest
import torch
from transformers import AutoTokenizer

from areal.api.alloc_mode import AllocationMode
from areal.api.io_struct import FinetuneSpec, SaveLoadMeta
from areal.experimental.api.cli_args import (
    ExperimentalTrainEngineConfig as TrainEngineConfig,
)
from areal.experimental.api.cli_args import (
    MegatronEngineConfig,
    OptimizerConfig,
)
from areal.experimental.megatron_engine import MegatronEngine
from areal.platforms import current_platform
from areal.utils import logging
from areal.utils.device import log_gpu_stats

logger = logging.getLogger("MegatronEngine Test")

VOCAB_SIZE = 100
MODEL_PATH = "/storage/openpsi/models/Qwen__Qwen3-0.6B/"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def mock_input(
    batch_size=5,
    min_seqlen=10,
    max_seqlen=20,
    device=current_platform.device_type,
) -> Dict[str, Any]:
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


def mock_loss_fn(logits: torch.Tensor, input_data: dict) -> torch.Tensor:
    """Mock loss function for testing."""
    return torch.mean(logits)


@pytest.fixture(scope="module")
def engine():
    logger.info(f"megatron.core version={get_version('megatron.core')}")
    os.environ.update(
        {
            "WORLD_SIZE": "1",
            "RANK": "0",
            "LOCAL_RANK": "0",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "7777",
        }
    )
    config = TrainEngineConfig(
        experiment_name="test",
        trial_name="test",
        path=MODEL_PATH,
        optimizer=OptimizerConfig(),
        megatron=MegatronEngineConfig(),
    )
    alloc_mode = AllocationMode.from_str("d1p1t1")
    ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=128, train_batch_size=8)
    engine = MegatronEngine(config)
    engine.create_process_group(alloc_mode.train)
    engine.initialize(addr=None, ft_spec=ft_spec, parallel_strategy=alloc_mode.train)
    logger.info(f"mcore GPTModel initialized: {engine.model}")
    log_gpu_stats("initialize")
    yield engine


def test_simple_forward(engine, mock_input):
    engine.eval()
    result = engine.forward(mock_input)
    logger.info(f"Forward done, result: {result}")


def test_simple_train(engine, mock_input):
    engine.train()
    train_result = engine.train_batch(
        mock_input, loss_fn=mock_loss_fn, loss_weight_fn=lambda x: 1
    )
    engine.step_lr_scheduler()
    logger.info(f"Train done, result={train_result}")


@torch.no_grad()
def test_hf_save_load_weights(tmp_path_factory, engine, mock_input):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    path = tmp_path_factory.mktemp("hf_engine_test")
    save_load_meta = SaveLoadMeta(
        path=path,
        weight_format="hf",
        tokenizer=tokenizer,
        with_optim=False,
        base_model_path=None,
    )

    old = engine.forward(input_=mock_input)
    start = time.perf_counter()
    engine.save(save_load_meta)
    logger.info(f"Save done, time cost: {time.perf_counter() - start:.4f} seconds.")
    for name, param in engine.model.named_parameters():
        param.zero_()

    start = time.perf_counter()
    engine.load(save_load_meta)
    logger.info(f"Load done, time cost: {time.perf_counter() - start:.4f} seconds.")
    new = engine.forward(input_=mock_input)
    assert torch.allclose(old, new)


@torch.no_grad()
def test_dcp_save_load_weights(tmp_path_factory, engine, mock_input):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    path = tmp_path_factory.mktemp("megatron_engine_dcp_test")
    save_load_meta = SaveLoadMeta(
        path=path,
        weight_format="dcp",
        tokenizer=tokenizer,
        with_optim=True,
        base_model_path=None,
    )

    old = engine.forward(input_=mock_input)
    start = time.perf_counter()
    engine.save(save_load_meta)
    logger.info(f"Save done, time cost: {time.perf_counter() - start:.4f} seconds.")
    for name, param in engine.model.named_parameters():
        param.zero_()

    start = time.perf_counter()
    engine.load(save_load_meta)
    logger.info(f"Load done, time cost: {time.perf_counter() - start:.4f} seconds.")
    new = engine.forward(input_=mock_input)
    assert torch.allclose(old, new)
