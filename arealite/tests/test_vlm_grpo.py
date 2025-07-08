"""Test script for GRPO Trainer implementation."""

import pytest
from datasets import load_dataset

from arealite.api.cli_args import (
    DatasetConfig,
    EngineBackendConfig,
    EngineConfig,
    GRPOTrainerConfig,
    OptimizerConfig,
    RLVRConfig,
    TrainerConfig,
    TrainingArgs,
)
from arealite.api.io_struct import FinetuneSpec
from arealite.api.rollout_api import RolloutCollectorFactory
from arealite.impl.trainer.grpo import SpmdGRPOTrainer
from arealite.system.rollout_controller import RolloutController
from arealite.tests.utils import mock_rollout_output
from realhf.base import constants, name_resolve, seeding
from realhf.api.core.data_api import load_hf_processor_and_tokenizer
from arealite.impl.dataset.VL_dataset import VLDataset
from arealite.api.trainer_api import TrainerFactory
EXPR_NAME = "test_vlm_grpo"
TRIAL_NAME = "test_vlm_grpo"
MODEL_PATH = "/storage/openpsi/models/Qwen2-VL-7B"

def create_vl_dataset(cfg: DatasetConfig, model_name_or_path: str) -> VLDataset:
    processor, tokenizer = load_hf_processor_and_tokenizer(
        model_name_or_path=model_name_or_path,
    )
    train_dataset = VLDataset(
        data_path=cfg.path,
        tokenizer=tokenizer,
        processor=processor,
        max_prompt_length=cfg.max_prompt_length,
        truncation="right",
        format_prompt=cfg.format_prompt,
        min_pixels=cfg.min_pixels,
        max_pixels=cfg.max_pixels,
        filter_overlong_prompts=False,
        filter_overlong_prompts_workers=cfg.filter_overlong_prompts_workers,
    )
    return train_dataset
@pytest.fixture(scope="module")
def args():
    args = TrainingArgs(experiment_name=EXPR_NAME, trial_name=TRIAL_NAME)
    constants.set_experiment_trial_names(args.experiment_name, args.trial_name)
    seeding.set_random_seed(args.seed, EXPR_NAME)
    args.train_dataset = DatasetConfig(
        path="/storage/openpsi/data/clevr_count_70k/",
        split="train",
        batch_size=4,
        shuffle=True,
        pin_memory=True,
        num_workers=1,
    )
    args.trainer = TrainerConfig(type="vl_grpo", grpo=GRPOTrainerConfig())
    args.trainer.grpo.actor = EngineConfig(
        path=MODEL_PATH,
        gradient_checkpointing=False,
        optimizer=OptimizerConfig(),
        backend=EngineBackendConfig(type="hf"),
    )
    args.trainer.grpo.ref = EngineConfig(
        path=MODEL_PATH,
        gradient_checkpointing=False,
        backend=EngineBackendConfig(type="hf"),
    )
    args.rollout.model_path = MODEL_PATH
    args.rollout.server_backend = "sglang"
    args.rollout.collector.rlvr = RLVRConfig(solution_path="nothing")
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
    args.trainer.grpo.kl_ctl = kl_ctl
    args.trainer.grpo.recompute_logprobs = recompute
    args.trainer.grpo.use_decoupled_loss = use_decoupled_loss
    args.train_dataset.batch_size = bs
    args.rollout.collector.rlvr.reward_type = "clevr_count_70k"
    # Create mock rollout controller and trainer
    rollout_factory = RolloutCollectorFactory(args)
    collector = rollout_factory.make_collector(args.rollout.collector)
    rollout_controller = RolloutController(args, args.rollout, collector=collector)
    dataset = create_vl_dataset(args.train_dataset, MODEL_PATH)

    if args.trainer is not None:
        trainer_factory = TrainerFactory(args)
        trainer = trainer_factory.make_trainer(
            args.trainer,
            train_dataset=dataset,
            valid_dataset=None,
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
    assert len(stats_list) == args.trainer.grpo.ppo_n_minibatches
    for stats in stats_list:
        assert isinstance(stats, dict)
        for k, v in stats.items():
            assert isinstance(v, float)
