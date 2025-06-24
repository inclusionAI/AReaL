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
    PPOTrainerConfig,
    RolloutControllerConfig,
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


def create_mock_input(bs: int = 2, min_seqlen: int = 3, max_seqlen: int = 12) -> Dict:
    """Create mock input data for testing."""
    seqlens = torch.randint(
        min_seqlen, max_seqlen, (bs,), dtype=torch.int, device="cuda"
    )
    max_seqlen = int(max(seqlens))
    input_ids = torch.randint(0, 100, (bs, max_seqlen), dtype=torch.long, device="cuda")

    attn_mask = torch.zeros((bs, max_seqlen), dtype=torch.bool, device="cuda")
    attn_mask[
        torch.arange(0, max_seqlen, device="cuda").unsqueeze(0) < seqlens.unsqueeze(1)
    ] = 1

    packed_input_ids, indices, cu_seqlens, max_seqlen = unpad_input(
        input_ids, attn_mask
    )

    assert torch.allclose(
        cu_seqlens, torch.nn.functional.pad(seqlens.cumsum(0, dtype=torch.int), (1, 0))
    )
    position_ids = compute_varlen_position_indices(int(sum(seqlens)), cu_seqlens)

    return dict(
        input_ids=packed_input_ids.unsqueeze(0),
        attention_mask=None,
        position_ids=position_ids.unsqueeze(0),
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        use_cache=False,
    )


def mock_loss_fn(logits: torch.Tensor, input_data: Dict) -> torch.Tensor:
    """Mock loss function for testing."""
    return torch.mean(logits)


def mock_loss_weight_fn(logits: torch.Tensor, input_data: Dict) -> float:
    """Mock loss weight function for testing."""
    return float(input_data["attention_mask"].sum())


def create_dataset(cfg: DatasetConfig):
    # select five data for test
    dataset = load_dataset(
        cfg.path,
        name=cfg.name,
        split=cfg.split,
    )
    dataset = dataset.select(range(5))
    return dataset


def test_engine():
    """Test engine creation and basic functionality."""
    print("Testing PPO train creation...")

    train_dataset = DatasetConfig(
        path="openai/gsm8k",
        name="main",
        split="train",
        batch_size=8,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
    )

    valid_dataset = DatasetConfig(
        path="openai/gsm8k",
        name="main",
        split="test",
        batch_size=8,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
    )

    engine_config = EngineConfig(
        type=ModelFamily("qwen2", False),
        path="Qwen/Qwen2.5-0.5B-Instruct",
        gradient_checkpointing=False,
        optimizer=OptimizerConfig(),
        backend=EngineBackendConfig(type="hf"),
    )

    sglang_client_config = LLMClientConfig(
        server_backend="sglang",
        tokenizer_path="Qwen/Qwen2.5-0.5B-Instruct",
    )

    ppo_config = PPOTrainerConfig(
        actor=engine_config,
    )

    train_config = TrainerConfig(
        type="ppo",
        ppo=ppo_config,
    )

    rollout_controller_config = RolloutControllerConfig(
        llm_client=sglang_client_config,
    )

    args = TrainingArgs(
        mode="local",
        n_nodes=1,
        n_gpus_per_node=1,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        trainer=train_config,
        rollout=rollout_controller_config,
    )

    rollout_controller = None
    if args.rollout is not None:
        rollout_controller = RolloutController(args, args.rollout)
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
