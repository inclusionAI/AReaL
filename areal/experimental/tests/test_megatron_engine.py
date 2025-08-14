import os
from importlib.metadata import version as get_version

import pytest

from areal.api.cli_args import TrainEngineConfig
from areal.api.io_struct import FinetuneSpec
from areal.experimental.megatron_engine import MegatronEngine
from areal.utils.device import log_gpu_stats
from realhf.base import logging

logger = logging.getLogger("MegatronEngine Test")

MODEL_PATH = "/storage/testing/models/Qwen__Qwen3-1.7B/"


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
    )
    ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=128, train_batch_size=8)
    engine = MegatronEngine(config)
    engine.initialize(addr=None, ft_spec=ft_spec)
    logger.info(f"mcore GPTModel initialized: {engine.model}")
    log_gpu_stats("initialize")
    yield engine


def test_make_engine(engine):
    pass
