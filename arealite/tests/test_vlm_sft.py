"""Test script for FSDP Engine implementation."""

from typing import Dict
import os
import torch
from datasets import load_dataset, Image

from arealite.api.cli_args import (
    DatasetConfig,
    EngineBackendConfig,
    DatasetPreprocessor,
    EngineConfig,
    OptimizerConfig,
    SFTTrainerConfig,
    TrainerConfig,
    TrainingArgs,
)
from arealite.api.trainer_api import TrainerFactory
from arealite.api.dataset_api import Multimodal_DatasetFactory
from arealite.impl.dataset.VL_dataset import VLDataset
from realhf.api.core.data_api import load_hf_processor_and_tokenizer
def mock_loss_fn(logits: torch.Tensor, input_data: Dict) -> torch.Tensor:
    """Mock loss function for testing."""
    return torch.mean(logits)


def mock_loss_weight_fn(logits: torch.Tensor, input_data: Dict) -> float:
    """Mock loss weight function for testing."""
    return float(input_data["attention_mask"].sum())


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


def test_sft():
    """Test engine creation and basic functionality."""
    # breakpoint()
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "7777"

    train_dataset = DatasetConfig(
        path="/storage/openpsi/data/clevr_count_70k/",
        preprocessor=DatasetPreprocessor("clevr_count_70k_sft"),
        name="default",
        split="train",
        batch_size=8,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
    )

    # valid_dataset = DatasetConfig(
    #     path="/storage/openpsi/data/clevr_count_70k/",
    #     name="main",
    #     split="test",
    #     batch_size=8,
    #     shuffle=False,
    #     pin_memory=True,
    #     num_workers=4,
    # )

    engine_config = EngineConfig(
        path="/storage/openpsi/models/Qwen2-VL-7B",
        gradient_checkpointing=False,
        optimizer=OptimizerConfig(),
        backend=EngineBackendConfig(type="hf"),
    )

    sft_config = SFTTrainerConfig(
        model=engine_config,
    )

    train_config = TrainerConfig(
        type="vl_sft",
        sft=sft_config,
    )

    args = TrainingArgs(
        experiment_name="vlm-test-sft",
        trial_name="test",
        mode="local",
        n_nodes=1,
        n_gpus_per_node=8,
        train_dataset=train_dataset,
        trainer=train_config,
    )

    rollout_controller = None
    dataset_factory =Multimodal_DatasetFactory(args)
    train_dataset = dataset_factory.make_dataset(args.train_dataset, 0, 1)
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
test_sft()