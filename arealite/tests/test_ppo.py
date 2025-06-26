"""Test script for PPO Trainer implementation."""

import random
from typing import Dict

import pytest
import torch
from datasets import load_dataset

from arealite.api.cli_args import (
    DatasetConfig,
    EngineBackendConfig,
    EngineConfig,
    FSDPConfig,
    LLMClientConfig,
    MicroBatchSpec,
    ModelFamily,
    OptimizerConfig,
    PPOTrainerConfig,
    RLVRConfig,
    RolloutControllerConfig,
    TrainerConfig,
    TrainingArgs,
)
from arealite.api.engine_api import EngineFactory
from arealite.api.io_struct import FinetuneSpec, Trajectory, TrajStats
from arealite.api.rollout_api import RolloutWorkflowFactory
from arealite.api.trainer_api import TrainerFactory
from arealite.impl.rollout_controller import RolloutController
from arealite.impl.trainer.ppo import SpmdPPOTrainer
from arealite.tests.utils import mock_rollout_output
from arealite.utils import (
    compute_varlen_position_indices,
    split_dict_tensor_with_cu_seqlens,
)
from realhf.base import constants, name_resolve, seeding
from realhf.impl.model.utils.padding import unpad_input

EXPR_NAME = "test_ppo"
TRIAL_NAME = "test_ppo"
MODEL_PATH = "Qwen/Qwen2.5-0.5B"


@pytest.fixture(scope="module")
def args():
    args = TrainingArgs(experiment_name=EXPR_NAME, trial_name=TRIAL_NAME)
    constants.set_experiment_trial_names(args.experiment_name, args.trial_name)
    seeding.set_random_seed(args.seed, EXPR_NAME)
    args.rollout.llm_client.tokenizer_path = MODEL_PATH
    args.train_dataset = DatasetConfig(
        path="openai/gsm8k",
        name="main",
        split="train",
        batch_size=4,
        shuffle=True,
        pin_memory=True,
        num_workers=1,
    )
    args.trainer = TrainerConfig(type="ppo", ppo=PPOTrainerConfig())
    args.trainer.ppo.actor = EngineConfig(
        type=ModelFamily("qwen2", False),
        path=MODEL_PATH,
        gradient_checkpointing=False,
        optimizer=OptimizerConfig(),
        backend=EngineBackendConfig(type="hf"),
    )
    args.trainer.ppo.ref = EngineConfig(
        type=ModelFamily("qwen2", False),
        path=MODEL_PATH,
        gradient_checkpointing=False,
        backend=EngineBackendConfig(type="hf"),
    )
    args.rollout.llm_client = LLMClientConfig(
        server_backend="sglang",
        tokenizer_path=MODEL_PATH,
    )
    args.rollout.workflow.rlvr = RLVRConfig(solution_path="nothing")
    args.rollout.gconfig.max_new_tokens = 16
    name_resolve.reconfigure(args.cluster.name_resolve)
    yield args
    name_resolve.reset()


@pytest.mark.parametrize("kl_ctl", [0.0, 0.1])
@pytest.mark.parametrize("bs", [4])
@pytest.mark.parametrize("n_samples", [2])
@pytest.mark.parametrize("recompute", [False, True])
@pytest.mark.parametrize("use_decoupled_loss", [False, True])
def test_train_step(args, kl_ctl, bs, n_samples, recompute, use_decoupled_loss):
    args.rollout.gconfig.n_samples = n_samples
    args.trainer.ppo.kl_ctl = kl_ctl
    args.trainer.ppo.recompute_logprobs = recompute
    args.trainer.ppo.use_decoupled_loss = use_decoupled_loss
    args.train_dataset.batch_size = bs
    # Create mock rollout controller and trainer
    rollout_factory = RolloutWorkflowFactory(args)
    workflow = rollout_factory.make_workflow(args.rollout.workflow)
    rollout_controller = RolloutController(args, args.rollout, workflow=workflow)
    dataset = load_dataset("openai/gsm8k", name="main", split="train").select(range(10))

    trainer = SpmdPPOTrainer(
        args=args,
        trainer_config=args.trainer,
        train_dataset=dataset,
        rollout_controller=rollout_controller,
    )
    ft_spec = FinetuneSpec(
        total_train_epochs=1,
        dataset_size=100,
        train_batch_size=args.train_dataset.batch_size,
    )
    trainer.actor.init_distributed(None, ft_spec)
    trainer.actor.eval()
    if trainer.ref is not None:
        trainer.ref.init_distributed(None, ft_spec)
        trainer.ref.eval()

    rollout_output = mock_rollout_output(bs, n_samples)
    stats_list = trainer._train_step(rollout_output)

    # Verify the output
    assert isinstance(stats_list, list)
    assert len(stats_list) == args.trainer.ppo.ppo_n_minibatches
    for stats in stats_list:
        assert isinstance(stats, dict)
        for k, v in stats.items():
            assert isinstance(v, float)
