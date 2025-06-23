# Copyright 2025 Ant Group Inc.
# Licensed under the Apache License, Version 2.0

"""Test script for HF Engine implementation."""

import os
from typing import Dict

import torch
from transformers import AutoTokenizer

from arealite.api.cli_args import (
    EngineBackendConfig,
    EngineConfig,
    HFConfig,
    MicroBatchSpec,
    ModelFamily,
    OptimizerConfig,
    TrainingArgs
)

from arealite.api.engine_api import EngineFactory
from arealite.api.io_struct import FinetuneSpec
from arealite.utils import (
    compute_varlen_position_indices,
    split_dict_tensor_with_cu_seqlens,
)
from realhf.impl.model.utils.padding import unpad_input


def create_mock_input(bs: int = 2, min_seqlen: int = 3, max_seqlen: int = 12) -> Dict:
    """Create mock input data for testing."""
    seqlens = torch.randint(
        min_seqlen, max_seqlen, (bs,), dtype=torch.int, device="cuda:0"
    )
    max_seqlen = int(max(seqlens))
    input_ids = torch.randint(0, 100, (bs, max_seqlen), dtype=torch.long, device="cuda:0")

    attn_mask = torch.zeros((bs, max_seqlen), dtype=torch.bool, device="cuda:0")
    attn_mask[
        torch.arange(0, max_seqlen, device="cuda:0").unsqueeze(0) < seqlens.unsqueeze(1)
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


def test_hf_engine():
    """Test engine creation and basic functionality."""
    print("Testing HF Engine creation...")
    
    engine_config = EngineConfig(
        type=ModelFamily("qwen2", False),
        path="Qwen/Qwen2.5-0.5B-Instruct",
        gradient_checkpointing=False,
        optimizer=OptimizerConfig(),
        backend=EngineBackendConfig(type="hf", hf=HFConfig(
            device="cuda:0",
        ))
    )

    mock_args = TrainingArgs(
        n_nodes=1, n_gpus_per_node=1
    )
    
    engine_factory = EngineFactory(mock_args)
    engine = engine_factory.make_engine(engine_config)
    ft_spec = FinetuneSpec(
        total_train_epochs=1, dataset_size=100, train_batch_size=2
    )
    engine.init_distributed(None, ft_spec)
    print("✓ Engine created successfully")

    print("Testing forward pass...")
    input_data = create_mock_input(bs=4)
    mb_spec = MicroBatchSpec(n_mbs=2)
    
    def simple_post_hook(logits, inp):
        return logits.shape

    result = engine.forward(
        input_=input_data,
        mb_spec=mb_spec,
        post_hook=simple_post_hook,
        aggregate_fn=lambda x: x[0] if x else None,
    )
    print(f"✓ Forward pass successful, output shape: {result}")

    print("Testing evaluation...")
    eval_result = engine.eval_batch(
        input_=input_data, mb_spec=mb_spec, loss_fn=mock_loss_fn
    )

    if eval_result is not None:
        print(f"✓ Evaluation successful, loss: {eval_result}")
    else:
        print(
            "✓ Evaluation completed (no loss returned - expected for non-final pipeline stages)"
        )
    
    print("Testing train ...")
    train_result = engine.train_batch(
        input_=input_data,
        mb_spec=mb_spec,
        loss_fn=mock_loss_fn,
        loss_weight_fn=lambda x: x["cu_seqlens"][-1],
        version_steps=0,
    )
    print(f"✓ Train successful")

    print("Testing get_hf_model_state_dict ...")
    model_dict = engine.get_hf_model_state_dict()
    print(f"✓ Model state dict retrieved successfully, dict is {model_dict}")

    print("Testing save_model_to_hf ...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    engine.save_model_to_hf(tokenizer=tokenizer, path="test_model")
    print("✓ Model saved successfully")

    print("Testing load_model_from_hf ...")
    engine.load_model_from_hf("Qwen/Qwen2.5-0.5B-Instruct")
    print("✓ Model loaded successfully")

    print("Testing save_optimizer_state ...")
    engine.save_optimizer_state("test_optimizer")
    print("✓ Optimizer saved successfully")

    print("Testing load_optimizer_state ...")
    engine.load_optimizer_state("test_optimizer")
    print("✓ Optimizer loaded successfully")

    print("All tests passed!")

