import asyncio
import threading
import time
from copy import deepcopy
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
import torch.distributed as dist
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchdata.stateful_dataloader import StatefulDataLoader

from arealite.api.cli_args import (
    GenerationHyperparameters,
    LLMServiceConfig,
    RLVRConfig,
    RolloutControllerConfig,
    SGLangConfig,
    TrainingArgs,
)
from arealite.api.io_struct import Trajectory, TrajStats
from arealite.api.llm_server_api import LLMServerFactory
from arealite.api.rollout_api import RolloutWorkflowFactory
from arealite.impl.rollout_controller import RolloutController
from arealite.utils import (
    concat_padded_tensors,
    list_of_dict2dict_of_list,
    pad_sequences_to_tensors,
)
from realhf.api.core.data_api import load_hf_tokenizer
from realhf.base import constants, name_resolve, seeding

EXPR_NAME = "test_rollout_controller"
TRIAL_NAME = "test_rollout_controller"
MODEL_PATH = "/storage/testing/models/Qwen__Qwen3-1.7B/"


@pytest.fixture(scope="module")
def args():
    args = TrainingArgs(experiment_name=EXPR_NAME, trial_name=TRIAL_NAME)
    constants.set_experiment_trial_names(args.experiment_name, args.trial_name)
    seeding.set_random_seed(args.seed, EXPR_NAME)
    args.rollout.llm_client.tokenizer_path = MODEL_PATH
    args.rollout.workflow.rlvr = RLVRConfig(
        solution_path=str(Path(__file__).parent / "data" / f"rlvr_math_dataset.jsonl")
    )
    name_resolve.reconfigure(args.cluster.name_resolve)
    yield args
    name_resolve.reset()


@pytest.fixture(scope="module")
def sglang_server(args):
    server_args = LLMServiceConfig(EXPR_NAME, TRIAL_NAME, model_path=MODEL_PATH)
    server_args.sglang = SGLangConfig()
    server = LLMServerFactory.make_server(server_args)
    server._startup()
    yield
    server._graceful_exit(0)


@pytest.fixture
def dataloader(args):
    dataset = load_dataset(
        "json",
        split="train",
        data_files=str(Path(__file__).parent / "data" / f"rlvr_math_dataset.jsonl"),
    )
    tokenizer = load_hf_tokenizer(MODEL_PATH)
    dataset = dataset.map(lambda x: tokenizer(x["prompt"]), batched=True)
    yield StatefulDataLoader(
        dataset,
        batch_size=2,
        collate_fn=lambda x: x,
    )


@pytest.mark.parametrize("num_workers", [1, 4])
@pytest.mark.parametrize("n_samples", [1, 2])
def test_generate_batch(args, sglang_server, dataloader, n_samples, num_workers):
    args = deepcopy(args)
    args.rollout.num_workers = num_workers
    args.rollout.gconfig.n_samples = n_samples
    rollout_factory = RolloutWorkflowFactory(args)
    workflow = rollout_factory.make_workflow(args.rollout.workflow)
    rollout_controller = RolloutController(args, args.rollout, workflow=workflow)

    data = next(iter(dataloader))
    batch_size = len(data)
    result = rollout_controller.generate_batch(batch_size, env_options=data)

    assert len(result) == batch_size * n_samples
    assert all(isinstance(traj, Trajectory) for traj in result)
    for traj in result:
        shape = traj.data["input_ids"].shape
        for v in traj.data.values():
            assert v.shape == shape
    data = pad_sequences_to_tensors([traj.data for traj in result])
    assert data["input_ids"].shape[0] == batch_size * n_samples
    shape = data["input_ids"].shape
    for v in data.values():
        assert v.shape == shape


# @patch("arealite.impl.rollout_controller.ProcessPoolExecutor")
# @patch("arealite.impl.rollout_controller.RolloutWorkflowFactory")
# def test_generate_batch_parallel(
#     mock_factory, mock_executor, rollout_controller, mock_trajectory
# ):
#     rollout_controller.config.num_workers = 2

#     mock_workflow_instance = Mock()
#     mock_factory.return_value.make_workflow.return_value = mock_workflow_instance

#     mock_executor_instance = Mock()
#     mock_executor.return_value.__enter__.return_value = mock_executor_instance
#     mock_executor_instance.map.return_value = [mock_trajectory, mock_trajectory]

#     batch_size = 2
#     result = rollout_controller.generate_batch(batch_size)

#     assert len(result) == batch_size
#     mock_executor.assert_called_once()
#     mock_executor_instance.map.assert_called_once()


# def test_set_version(rollout_controller):
#     new_version = 5
#     rollout_controller.set_version(new_version)

#     assert rollout_controller._version == new_version


# def test_start_stop_generate_loop(rollout_controller):
#     mock_dataloader = Mock(spec=DataLoader)

#     with patch.object(rollout_controller, "_generate_until_complete") as mock_generate:
#         rollout_controller.start_generate_loop(mock_dataloader)

#         assert hasattr(rollout_controller, "_generation_thread")
#         assert rollout_controller._generation_thread.is_alive()

#         rollout_controller.stop_generate_loop()

#         assert rollout_controller._exiting.is_set()
#         assert not rollout_controller._generation_thread.is_alive()
#         mock_generate.assert_called_once_with(mock_dataloader)


# def test_prepare_batch_empty_buffer(rollout_controller):
#     batch_size = 2

#     with patch.object(rollout_controller, "_prepare_batch_async") as mock_async:
#         mock_async.return_value = []

#         result = rollout_controller.prepare_batch(batch_size)

#         mock_async.assert_called_once_with(batch_size)
#         assert result == []


# def test_prepare_batch_with_buffer(rollout_controller, mock_trajectory):
#     rollout_controller._buffer = [[mock_trajectory], [mock_trajectory]]
#     batch_size = 1

#     with patch("arealite.impl.rollout_controller.datapack.flat2d") as mock_flat2d:
#         mock_flat2d.return_value = [mock_trajectory]

#         result = rollout_controller.prepare_batch(batch_size)

#         mock_flat2d.assert_called_once()
#         assert len(rollout_controller._buffer) == 1


# @patch("torch.distributed.get_world_size")
# async def test_generate_loop_basic(
#     mock_world_size, rollout_controller, mock_trajectory
# ):
#     mock_world_size.return_value = 1

#     dataset = MockDataset(size=2)
#     dataloader = DataLoader(dataset, batch_size=1)

#     rollout_controller.workflow.run_episode_async.return_value = mock_trajectory

#     # Set exiting flag after short delay to terminate loop
#     def set_exit():
#         time.sleep(0.1)
#         rollout_controller._exiting.set()

#     exit_thread = threading.Thread(target=set_exit)
#     exit_thread.start()

#     await rollout_controller._generate_loop(dataloader)

#     exit_thread.join()


# def test_generate_until_complete(rollout_controller):
#     mock_dataloader = Mock(spec=DataLoader)

#     with patch.object(rollout_controller, "_generate_loop") as mock_loop:
#         mock_loop.return_value = asyncio.sleep(0)  # Mock async function

#         # Set exit flag to prevent infinite loop
#         rollout_controller._exiting.set()

#         rollout_controller._generate_until_complete(mock_dataloader)

#         mock_loop.assert_called_once_with(mock_dataloader)
