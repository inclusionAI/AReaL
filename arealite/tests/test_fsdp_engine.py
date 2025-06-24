"""Test script for FSDP Engine implementation."""

import os
from typing import Dict

import torch

from arealite.api.cli_args import (
    EngineBackendConfig,
    EngineConfig,
    FSDPConfig,
    MicroBatchSpec,
    ModelFamily,
    OptimizerConfig,
)
from arealite.api.engine_api import EngineFactory
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


def test_split_mbs():
    # dist.init_process_group(backend="nccl")
    torch.cuda.set_device(0)

    def print_data(data):
        for k, v in data.items():
            print(f"k={k}")
            if torch.is_tensor(v):
                print(f"v.shape={v.shape}")
            print(f"v={v}")

    input_data = create_mock_input(bs=4)
    print("***** full data *****")
    print_data(input_data)

    mb_spec = MicroBatchSpec(n_mbs=2)
    mbs = split_dict_tensor_with_cu_seqlens(input_data, mb_spec)
    for i, mb in enumerate(mbs):
        print(f"***** data batch {i} *****")
        print_data(mb)

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

def test_hf_forward():
    from transformers import AutoModelForCausalLM
    path = "/storage/openpsi/models/Qwen__Qwen2.5-0.5B-Instruct"
    # path = "/storage/openpsi/models/Qwen__Qwen3-1.7B"
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=path,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    model = model.to("cuda")

    data = create_mock_input(bs=1, min_seqlen=100, max_seqlen=200)
    r = model(**data)

    print(r.logits.shape)

def test_engine():
    """Test engine creation and basic functionality."""
    print("Testing FSDP Engine creation...")
    # init torch dist
    try:
        # Note: This test needs TrainingArgs, but we're using a mock config
        # In a real scenario, you'd need to provide proper TrainingArgs
        from arealite.api.cli_args import TrainingArgs
        from arealite.api.io_struct import FinetuneSpec

        config = EngineConfig(
            type=ModelFamily("qwen2", False),
            path="/storage/openpsi/models/Qwen__Qwen2.5-0.5B-Instruct",
            gradient_checkpointing=False,
            optimizer=OptimizerConfig(),
            backend=EngineBackendConfig(type="fsdp", fsdp=FSDPConfig()),
        )
        mock_args = TrainingArgs(
            n_nodes=1, n_gpus_per_node=1
        )  # This would need proper initialization
        engine_factory = EngineFactory(mock_args)
        engine = engine_factory.make_engine(config)
        ft_spec = FinetuneSpec(
            total_train_epochs=1, dataset_size=100, train_batch_size=2
        )
        engine.init_distributed(None, ft_spec)
        print("✓ Engine created successfully")

        # Test forward pass
        print("Testing forward pass...")
        input_data = create_mock_input(bs=1)
        mb_spec = MicroBatchSpec(n_mbs=1)

        def simple_post_hook(logits, inp):
            return logits.shape

        result = engine.forward(
            input_=input_data,
            mb_spec=mb_spec,
            post_hook=simple_post_hook,
            aggregate_fn=lambda x: x[0] if x else None,
        )

        print(f"✓ Forward pass successful, output shape: {result}")

        # Test evaluation
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
        )
        print(f"✓ Train successful")

        print("All tests passed!")

    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # Only run if we have GPU available (FSDP requires CUDA)
    # if torch.cuda.is_available():
    #     test_engine_creation()
    # else:
    #     print("CUDA not available, skipping FSDP engine test")
    # print("Engine implementation completed successfully!")
    # test_split_mbs()
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "7777"

    # test_engine()
    test_hf_forward()
