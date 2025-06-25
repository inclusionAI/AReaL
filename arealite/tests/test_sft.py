"""Test script for FSDP Engine implementation."""

import os
from typing import Dict

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
    RolloutControllerConfig,
    SFTTrainerConfig,
    TrainerConfig,
    TrainingArgs,
)
from arealite.api.engine_api import EngineFactory
from arealite.api.trainer_api import TrainerFactory
from arealite.impl.rollout_controller import RolloutController
from arealite.utils import (
    compute_varlen_position_indices,
    split_dict_tensor_with_cu_seqlens,
)
from realhf.impl.model.utils.padding import unpad_input


def mock_loss_fn(logits: torch.Tensor, input_data: Dict) -> torch.Tensor:
    """Mock loss function for testing."""
    return torch.mean(logits)


def mock_loss_weight_fn(logits: torch.Tensor, input_data: Dict) -> float:
    """Mock loss weight function for testing."""
    return float(input_data["attention_mask"].sum())


def create_dataset(cfg: DatasetConfig):
    dataset = load_dataset(
        cfg.path,
        name=cfg.name,
        split=cfg.split,
    )
    return dataset


def test_engine():
    """Test engine creation and basic functionality."""
    print("Testing PPO train creation...")

    train_dataset = DatasetConfig(
        path="/storage/openpsi/users/meizhiyu.mzy/datasets/sft/json/",
        # name="main",
        # split="train",
        batch_size=8,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
    )

    valid_dataset = DatasetConfig(
        path="/storage/openpsi/users/meizhiyu.mzy/datasets/sft/json/",
        # name="main",
        # split="test",
        batch_size=8,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
    )

    engine_config = EngineConfig(
        type=ModelFamily("qwen2", False),
        path="/storage/openpsi/models/Qwen__Qwen2.5-0.5B-Instruct/",
        gradient_checkpointing=False,
        optimizer=OptimizerConfig(),
        backend=EngineBackendConfig(type="hf"),
    )

    sft_config = SFTTrainerConfig(
        model=engine_config,
    )

    train_config = TrainerConfig(
        type="sft",
        sft=sft_config,
    )

    args = TrainingArgs(
        experiment_name="test-sft",
        trial_name="test",
        mode="local",
        n_nodes=1,
        n_gpus_per_node=1,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        trainer=train_config,
    )

    rollout_controller = None
    train_dataset = create_dataset(args.train_dataset)
    valid_dataset = None
    if args.valid_dataset is not None:
        valid_dataset = create_dataset(args.valid_dataset)
    if args.trainer is not None:
        trainer_factory = TrainerFactory(args)
        trainer = trainer_factory.make_trainer(
            args.trainer,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            rollout_controller=rollout_controller,
        )
        trainer.train()

    print("All tests passed!")
