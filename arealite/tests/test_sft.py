"""Test script for FSDP Engine implementation."""

from typing import Dict
import os

import torch
from datasets import load_dataset

from arealite.api.cli_args import (
    DatasetConfig,
    EngineBackendConfig,
    EngineConfig,
    ModelFamily,
    OptimizerConfig,
    SFTTrainerConfig,
    TrainerConfig,
    TrainingArgs,
    DatasetPreprocessor
)
from arealite.api.dataset_api import DatasetFactory
from arealite.api.trainer_api import TrainerFactory


def mock_loss_fn(logits: torch.Tensor, input_data: Dict) -> torch.Tensor:
    """Mock loss function for testing."""
    return torch.mean(logits)


def mock_loss_weight_fn(logits: torch.Tensor, input_data: Dict) -> float:
    """Mock loss weight function for testing."""
    return float(input_data["attention_mask"].sum())


def test_sft():
    """Test engine creation and basic functionality."""
    # environment variables for torch distributed
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "7777"

    train_dataset = DatasetConfig(
        path="openai/gsm8k",
        preprocessor=DatasetPreprocessor("gsm8k_sft"),
        name="main",
        split="train",
        batch_size=8,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
    )

    valid_dataset = DatasetConfig(
        path="openai/gsm8k",
        preprocessor=DatasetPreprocessor("gsm8k_sft"),
        name="main",
        split="test",
        batch_size=8,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
    )

    engine_config = EngineConfig(
        type=ModelFamily("qwen2", False),
        path="Qwen/Qwen2.5-0.5B",
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
    dataset_factory = DatasetFactory(args)
    train_dataset = dataset_factory.make_dataset(args.train_dataset, 0, 1)
    train_dataset = train_dataset.select(range(100))
    valid_dataset = None
    if args.valid_dataset is not None:
        valid_dataset = dataset_factory.make_dataset(args.valid_dataset, 0, 1)
        valid_dataset = valid_dataset.select(range(100))
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
