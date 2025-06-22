import time
from copy import deepcopy
from pathlib import Path

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
from realhf.base import constants, name_resolve, names, seeding

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
        drop_last=True,
    )


@pytest.mark.parametrize("num_workers", [1, 4])
@pytest.mark.parametrize("n_samples", [1, 2])
def test_generate_batch(args, sglang_server, dataloader, n_samples, num_workers):
    args = deepcopy(args)
    args.rollout.num_workers = num_workers
    args.rollout.gconfig.n_samples = n_samples
    args.rollout.gconfig.max_new_tokens = 16
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


@pytest.mark.parametrize("n_samples", [1, 4, 16])
@pytest.mark.parametrize("num_workers", [1, 2, 4])
def test_async_rollout(args, sglang_server, dataloader, n_samples, num_workers):
    args = deepcopy(args)
    args.rollout.gconfig.n_samples = n_samples
    args.rollout.gconfig.max_new_tokens = 16
    args.train_dataset.batch_size = 2
    args.rollout.max_concurrent_rollouts = 16
    args.rollout.num_workers = num_workers
    rollout_factory = RolloutWorkflowFactory(args)
    workflow = rollout_factory.make_workflow(args.rollout.workflow)
    rollout_controller = RolloutController(args, args.rollout, workflow=workflow)

    # start loop
    rollout_controller.start_generate_loop()
    assert hasattr(rollout_controller, "_collector_thread")
    assert rollout_controller._collector_thread.is_alive()

    # Submit data to workers
    data = next(iter(dataloader))
    rollout_controller.submit(data)

    # wait for batch
    batch_size = 2
    result = rollout_controller.prepare_batch(batch_size)
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

    # exit
    rollout_controller.stop_generate_loop()
    assert rollout_controller._exiting.is_set()
    assert not rollout_controller._collector_thread.is_alive()
    assert not rollout_controller._worker_processes


@pytest.mark.parametrize("ofp", [1, 2, 4, 16])
def test_async_staleness_control(args, sglang_server, dataloader, ofp):
    args = deepcopy(args)
    args.rollout.gconfig.n_samples = 2
    args.rollout.gconfig.max_new_tokens = 4
    args.rollout.max_head_offpolicyness = ofp
    args.train_dataset.batch_size = 2
    args.rollout.max_concurrent_rollouts = 100
    rollout_factory = RolloutWorkflowFactory(args)
    workflow = rollout_factory.make_workflow(args.rollout.workflow)
    rollout_controller = RolloutController(args, args.rollout, workflow=workflow)
    name = names.model_version(args.experiment_name, args.trial_name, "actor")
    name_resolve.add(name, str(0), replace=True)

    # start loop
    rollout_controller.start_generate_loop()
    batch_size = 2

    gen = iter(dataloader)
    rollout_controller.submit(next(gen))
    rollout_controller.submit(next(gen))
    # wait for some time
    time.sleep(15)
    assert len(rollout_controller._buffer) == min(
        batch_size * 2, batch_size * (ofp + 1)
    )

    # Update model version
    name = names.model_version(args.experiment_name, args.trial_name, "actor")
    name_resolve.add(name, str(1), replace=True)
    print("Updated model version", flush=True)

    # submit again
    rollout_controller.submit(next(gen))
    rollout_controller.submit(next(gen))
    # wait for some time
    time.sleep(15)
    assert len(rollout_controller._buffer) == min(
        batch_size * 4, batch_size * (ofp + 2)
    )

    # exit
    rollout_controller.stop_generate_loop()
