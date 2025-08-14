import os
from importlib.metadata import version as get_version

import pytest
import torch
from tensordict import TensorDict

from areal.api.io_struct import FinetuneSpec
from areal.experimental.api.cli_args import (
    ExperimentalTrainEngineConfig as TrainEngineConfig,
)
from areal.experimental.api.cli_args import MegatronEngineConfig, OptimizerConfig
from areal.experimental.megatron_engine import MegatronEngine
from areal.utils.device import log_gpu_stats
from realhf.base import logging

logger = logging.getLogger("MegatronEngine Test")

VOCAB_SIZE = 100
MODEL_PATH = "/storage/testing/models/Qwen__Qwen3-1.7B/"


@pytest.fixture(scope="module")
def mock_input(
    batch_size=5,
    min_seqlen=10,
    max_seqlen=20,
    device="cuda:0",
) -> TensorDict:
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


def mock_loss_fn(logits: torch.Tensor, input_data: TensorDict) -> torch.Tensor:
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
    ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=128, train_batch_size=8)
    engine = MegatronEngine(config)
    engine.initialize(addr=None, ft_spec=ft_spec)
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
        mock_input, loss_fn=mock_loss_fn, loss_weight_fn=None
    )
    engine.step_lr_scheduler()
    logger.info(f"Train done, result={train_result}")
