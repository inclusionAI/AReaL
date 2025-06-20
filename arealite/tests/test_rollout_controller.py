import asyncio
import threading
import time
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset

from arealite.api.cli_args import (
    GenerationHyperparameters,
    GlobalConfig,
    RolloutControllerConfig,
    TrainingArgs,
)
from arealite.api.io_struct import Trajectory, TrajStats
from arealite.impl.rollout_controller import RolloutController


class MockDataset(Dataset):
    def __init__(self, size=10):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {"data": f"sample_{idx}"}


@pytest.fixture
def mock_trajectory():
    stats = TrajStats(
        episode_length=1,
        total_reward=0.5,
        start_time=time.time(),
    )
    return Trajectory(
        prompt={"input_ids": [1, 2, 3]},
        data={"input_ids": [1, 2, 3, 4]},
        stats=stats,
    )


@pytest.fixture
def training_args():
    args = TrainingArgs(experiment_name="test", trial_name="test")
    args.train_dataset.batch_size = 4
    return args


@pytest.fixture
def controller_config():
    config = RolloutControllerConfig()
    config.num_workers = 1
    config.max_concurrent_rollouts = 8
    config.filter_reward_lb = 0.0
    config.filter_reward_ub = 1.0
    config.gconfig = GenerationHyperparameters(n_samples=1)
    return config


@pytest.fixture
def mock_workflow():
    workflow = Mock()
    workflow.run_episode = Mock()
    workflow.run_episode_async = AsyncMock()
    return workflow


@pytest.fixture
def rollout_controller(training_args, controller_config, mock_workflow):
    return RolloutController(training_args, controller_config, mock_workflow)


def test_init(training_args, controller_config, mock_workflow):
    controller = RolloutController(training_args, controller_config, mock_workflow)

    assert controller.args == training_args
    assert controller.config == controller_config
    assert controller.workflow == mock_workflow
    assert controller.train_batch_size == training_args.train_dataset.batch_size
    assert controller._version == 0
    assert controller._buffer == []


def test_generate_batch_sequential(rollout_controller, mock_trajectory):
    rollout_controller.workflow.run_episode.return_value = mock_trajectory

    batch_size = 2
    result = rollout_controller.generate_batch(batch_size)

    assert len(result) == batch_size
    assert all(isinstance(traj, Trajectory) for traj in result)
    assert rollout_controller.workflow.run_episode.call_count == batch_size


def test_generate_batch_with_env_options_and_seeds(rollout_controller, mock_trajectory):
    rollout_controller.workflow.run_episode.return_value = mock_trajectory

    batch_size = 2
    env_options = [{"opt1": "val1"}, {"opt2": "val2"}]
    seeds = [123, 456]

    result = rollout_controller.generate_batch(batch_size, env_options, seeds)

    assert len(result) == batch_size
    calls = rollout_controller.workflow.run_episode.call_args_list
    assert len(calls) == batch_size


@patch("arealite.impl.rollout_controller.ProcessPoolExecutor")
@patch("arealite.impl.rollout_controller.RolloutWorkflowFactory")
def test_generate_batch_parallel(
    mock_factory, mock_executor, rollout_controller, mock_trajectory
):
    rollout_controller.config.num_workers = 2

    mock_workflow_instance = Mock()
    mock_factory.return_value.make_workflow.return_value = mock_workflow_instance

    mock_executor_instance = Mock()
    mock_executor.return_value.__enter__.return_value = mock_executor_instance
    mock_executor_instance.map.return_value = [mock_trajectory, mock_trajectory]

    batch_size = 2
    result = rollout_controller.generate_batch(batch_size)

    assert len(result) == batch_size
    mock_executor.assert_called_once()
    mock_executor_instance.map.assert_called_once()


def test_set_version(rollout_controller):
    new_version = 5
    rollout_controller.set_version(new_version)

    assert rollout_controller._version == new_version


def test_start_stop_generate_loop(rollout_controller):
    mock_dataloader = Mock(spec=DataLoader)

    with patch.object(rollout_controller, "_generate_until_complete") as mock_generate:
        rollout_controller.start_generate_loop(mock_dataloader)

        assert hasattr(rollout_controller, "_generation_thread")
        assert rollout_controller._generation_thread.is_alive()

        rollout_controller.stop_generate_loop()

        assert rollout_controller._exiting.is_set()
        assert not rollout_controller._generation_thread.is_alive()
        mock_generate.assert_called_once_with(mock_dataloader)


def test_prepare_batch_empty_buffer(rollout_controller):
    batch_size = 2

    with patch.object(rollout_controller, "_prepare_batch_async") as mock_async:
        mock_async.return_value = []

        result = rollout_controller.prepare_batch(batch_size)

        mock_async.assert_called_once_with(batch_size)
        assert result == []


def test_prepare_batch_with_buffer(rollout_controller, mock_trajectory):
    rollout_controller._buffer = [[mock_trajectory], [mock_trajectory]]
    batch_size = 1

    with patch("arealite.impl.rollout_controller.datapack.flat2d") as mock_flat2d:
        mock_flat2d.return_value = [mock_trajectory]

        result = rollout_controller.prepare_batch(batch_size)

        mock_flat2d.assert_called_once()
        assert len(rollout_controller._buffer) == 1


@patch("torch.distributed.get_world_size")
async def test_generate_loop_basic(
    mock_world_size, rollout_controller, mock_trajectory
):
    mock_world_size.return_value = 1

    dataset = MockDataset(size=2)
    dataloader = DataLoader(dataset, batch_size=1)

    rollout_controller.workflow.run_episode_async.return_value = mock_trajectory

    # Set exiting flag after short delay to terminate loop
    def set_exit():
        time.sleep(0.1)
        rollout_controller._exiting.set()

    exit_thread = threading.Thread(target=set_exit)
    exit_thread.start()

    await rollout_controller._generate_loop(dataloader)

    exit_thread.join()


def test_generate_until_complete(rollout_controller):
    mock_dataloader = Mock(spec=DataLoader)

    with patch.object(rollout_controller, "_generate_loop") as mock_loop:
        mock_loop.return_value = asyncio.sleep(0)  # Mock async function

        # Set exit flag to prevent infinite loop
        rollout_controller._exiting.set()

        rollout_controller._generate_until_complete(mock_dataloader)

        mock_loop.assert_called_once_with(mock_dataloader)
