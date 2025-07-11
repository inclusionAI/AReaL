import queue
import random
import time
from unittest.mock import patch

import pytest
import torch
from torch.utils.data import DataLoader

from realhf.api.cli_args import BaseExperimentConfig, NameResolveConfig
from realhf.api.core import config as config_api
from realhf.api.core import data_api
from realhf.base import constants, name_resolve, testing
from tests.fixtures import *


@pytest.fixture
def prompt_dataset_cfg(dataset, tokenizer):
    return [
        config_api.DatasetAbstraction(
            type_="prompt",
            args=dict(
                max_length=32,
                dataset_builder=lambda: dataset,
            ),
        )
    ]


class MockPuller:
    def __init__(self, *args, **kwargs):
        self.pull_count = 0

    def pull(self, timeout_ms: float):
        self.pull_count += 1
        if self.pull_count % 2 == 1:  # Return data every other call
            return [
                data_api.SequenceSample.as_json_compatible(
                    data_api.SequenceSample(
                        ids=[str(123)],
                        data={"packed_prompts": torch.tensor([1, 2, 3])},
                        keys=set(["packed_prompts"]),
                        seqlens=dict(packed_prompts=[[3]]),
                        dtypes=dict(packed_prompts=torch.long),
                        trailing_shapes=dict(packed_prompts=()),
                    )
                )
            ]
        raise queue.Empty()


@pytest.fixture
def mock_puller(monkeypatch):
    monkeypatch.setattr(
        "realhf.system.stream_dataset.NameResolvingZmqPuller", MockPuller
    )


def test_load_stream_dataset(prompt_dataset_cfg, tokenizer, mock_puller):
    import realhf.impl.dataset  # isort: skip
    from realhf.api.core.data_api import make_dataset
    from realhf.system.stream_dataset import PullerStreamDataset

    constants.set_experiment_trial_names(
        testing._DEFAULT_EXPR_NAME, testing._DEFAULT_TRIAL_NAME
    )

    name_resolve.reconfigure(NameResolveConfig("nfs", "/tmp/areal/test-stream-dataset"))
    testing.clear_name_resolve()

    util = data_api.DatasetUtility(
        seed=42,
        dp_rank=0,
        world_size=1,
        tokenizer=tokenizer,
    )

    # Test initialization
    dataset = PullerStreamDataset(
        util,
        args=BaseExperimentConfig(),
        dataset_cfgs=prompt_dataset_cfg,
        pull_timeout_ms=100,
    )
    assert len(dataset) > 0  # Should have non-zero size from prompt dataset
    assert dataset.data_queue.empty()

    # Test data pulling
    time.sleep(0.2)  # Give worker thread time to pull some data
    items = dataset[0]  # This should get all available items from queue
    assert len(items) > 0
    assert isinstance(items[0], data_api.SequenceSample)

    # Test queue behavior with multiple gets
    time.sleep(0.2)
    items1 = dataset[0]
    items2 = dataset[0]
    assert len(items1) + len(items2) > 0

    # Test cleanup
    dataset._stop_event.set()
    del dataset


def test_puller_stream_dataset_timeout(prompt_dataset_cfg, tokenizer):
    from realhf.system.stream_dataset import PullerStreamDataset

    name_resolve.reconfigure(NameResolveConfig("nfs", "/tmp/areal/test-stream-dataset"))
    testing.clear_name_resolve()

    util = data_api.DatasetUtility(
        seed=42,
        dp_rank=0,
        world_size=1,
        tokenizer=tokenizer,
    )

    with patch("realhf.system.stream_dataset.NameResolvingZmqPuller") as mock_puller:
        mock_puller.return_value.pull.side_effect = queue.Empty()

        dataset = PullerStreamDataset(
            util, BaseExperimentConfig(), prompt_dataset_cfg, pull_timeout_ms=10
        )
        # Should handle timeout gracefully
        assert dataset[0] == []
        dataset._stop_event.set()
        del dataset


def test_puller_stream_dataset_stop_event(prompt_dataset_cfg, tokenizer, mock_puller):
    from realhf.system.stream_dataset import PullerStreamDataset

    name_resolve.reconfigure(NameResolveConfig("nfs", "/tmp/areal/test-stream-dataset"))
    testing.clear_name_resolve()

    util = data_api.DatasetUtility(
        seed=42,
        dp_rank=0,
        world_size=1,
        tokenizer=tokenizer,
    )

    dataset = PullerStreamDataset(util, BaseExperimentConfig(), prompt_dataset_cfg)
    assert not dataset._stop_event.is_set()

    # Trigger stop event and verify thread stops
    dataset._stop_event.set()
    time.sleep(0.1)
    assert not dataset.worker_thread.is_alive()
    del dataset


def test_puller_stream_dataset_worker_thread_exception(prompt_dataset_cfg, tokenizer):
    from realhf.system.stream_dataset import PullerStreamDataset

    name_resolve.reconfigure(NameResolveConfig("nfs", "/tmp/areal/test-stream-dataset"))
    testing.clear_name_resolve()

    util = data_api.DatasetUtility(
        seed=42,
        dp_rank=0,
        world_size=1,
        tokenizer=tokenizer,
    )

    with patch("realhf.system.stream_dataset.NameResolvingZmqPuller") as mock_puller:
        mock_puller.return_value.pull.side_effect = Exception("Test error")

        dataset = PullerStreamDataset(util, BaseExperimentConfig(), prompt_dataset_cfg)
        time.sleep(0.1)  # Give thread time to crash
        assert not dataset.worker_thread.is_alive()
        del dataset
