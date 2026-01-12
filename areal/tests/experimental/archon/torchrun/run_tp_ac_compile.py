"""Test TP + AC + torch.compile compatibility.

This test verifies that _WaitAsyncWrapper prevents dynamo recompilation
when AsyncCollectiveTensor flows between checkpointed transformer blocks.

Run with torchrun (requires at least 2 GPUs for TP=2):
    torchrun --nproc_per_node=2 areal/tests/experimental/archon/torchrun/run_tp_ac_compile.py

This test would have raised ValueError before the fix due to the incompatible
combination of TP + AC + compile. After the fix with _WaitAsyncWrapper,
this combination should work without dynamo recompilation warnings.
"""

import argparse
import os
import warnings
from typing import Any

import torch
import torch.distributed as dist

from areal.api.alloc_mode import ParallelStrategy
from areal.api.cli_args import (
    ArchonEngineConfig,
    MicroBatchSpec,
    TrainEngineConfig,
)
from areal.api.io_struct import FinetuneSpec
from areal.experimental.engine.archon_engine import ArchonEngine
from areal.platforms import current_platform
from areal.tests.utils import get_model_path
from areal.utils.data import tensor_container_to

MODEL_PATHS = {
    "qwen3": get_model_path(
        "/storage/openpsi/models/Qwen__Qwen3-0.6B/", "Qwen/Qwen3-0.6B"
    ),
}


def setup_distributed_environment():
    """Set up distributed environment for torchrun."""
    if dist.is_initialized():
        return
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", "29500")

    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        world_size=world_size,
        rank=rank,
    )
    current_platform.set_device(rank)


def mock_input(
    device: torch.device,
    batch_size: int = 4,
    min_seqlen: int = 4,
    max_seqlen: int = 16,
) -> dict[str, Any]:
    """Create mock padded input data for testing."""
    pad_token_id = 0
    seqlens = torch.randint(
        min_seqlen, max_seqlen, (batch_size,), dtype=torch.int, device=device
    )
    max_len = int(max(seqlens))
    input_ids = torch.randint(
        10000, 50000, (batch_size, max_len), dtype=torch.long, device=device
    )
    attn_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)

    attn_mask[
        torch.arange(0, max_len, device=device).unsqueeze(0) < seqlens.unsqueeze(1)
    ] = 1
    input_ids.masked_fill_(~attn_mask, pad_token_id)

    return dict(
        input_ids=input_ids,
        attention_mask=attn_mask,
    )


def make_engine_with_tp_ac_compile(
    model_type: str, mb_spec: MicroBatchSpec, tp_size: int
):
    """Create and initialize a ArchonEngine with TP + AC + compile.

    This combination would have raised ValueError before the fix.
    """
    # Enable gradient_checkpointing (AC) and compile
    archon_config = ArchonEngineConfig(
        enable_compile=True,
        recompute_granularity="full",  # full AC
    )

    config = TrainEngineConfig(
        experiment_name="test_tp_ac_compile",
        trial_name="test",
        path=MODEL_PATHS[model_type],
        mb_spec=mb_spec,
        optimizer=None,  # No optimizer needed for forward-only test
        gradient_checkpointing=True,  # Enable AC
        archon=archon_config,
    )
    print(f"config = {config}")

    ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=128, train_batch_size=8)

    engine = ArchonEngine(config)

    # Create parallel strategy with TP
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    dp_size = world_size // tp_size
    parallel_strategy = ParallelStrategy(
        data_parallel_size=dp_size,
        tensor_parallel_size=tp_size,
    )

    engine.create_process_group(parallel_strategy=parallel_strategy)
    engine.initialize(addr=None, ft_spec=ft_spec)
    return engine


def test_tp_ac_compile_compatibility(model_type: str, tp_size: int):
    """Test TP + AC + compile compatibility.

    This test verifies:
    1. The combination of TP + AC + compile no longer raises ValueError
    2. No dynamo recompilation warnings are triggered during forward passes
    3. The model produces valid outputs
    """
    setup_distributed_environment()

    torch.manual_seed(42)

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    dp_size = world_size // tp_size

    print(
        f"[Rank {rank}] Testing TP + AC + compile compatibility with "
        f"world_size={world_size}, dp={dp_size}, tp={tp_size}"
    )

    if world_size < tp_size:
        print(f"[Rank {rank}] Skipping test: world_size < tp_size")
        return

    batch_size = 4
    mb_spec = MicroBatchSpec(n_mbs=2)

    # Capture warnings during engine creation and forward passes
    recompile_warnings = []

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Create engine with TP + AC + compile
        # This would have raised ValueError before the fix
        engine = make_engine_with_tp_ac_compile(model_type, mb_spec, tp_size)

        # Verify configuration
        assert engine.parallel_dims.tp_enabled, "TP should be enabled"
        assert engine.parallel_dims.tp == tp_size, f"Expected tp={tp_size}"
        print(
            f"[Rank {rank}] Engine initialized with "
            f"tp={engine.parallel_dims.tp}, dp={engine.parallel_dims.dp_shard}, "
            f"AC=enabled, compile=enabled"
        )

        # Create input
        if rank == 0:
            full_input = mock_input(
                device=torch.device(f"{current_platform.device_type}:0"),
                batch_size=batch_size,
                max_seqlen=16,
            )
            full_input_list = [full_input]
        else:
            full_input_list = [None]

        if dist.is_initialized():
            dist.broadcast_object_list(full_input_list, src=0, group=dist.group.WORLD)

        full_input = full_input_list[0]
        full_input = tensor_container_to(
            full_input, torch.device(f"{current_platform.device_type}:{rank}")
        )

        # Run multiple forward passes to trigger potential recompilation
        engine.eval()
        num_iterations = 5
        for i in range(num_iterations):
            logprobs = engine.forward_batch(
                input_=full_input,
                aggregate_fn=lambda xs: torch.cat(xs, dim=0),
            )
            if i == 0:
                print(f"[Rank {rank}] First forward pass completed")
                print(f"[Rank {rank}] Output logprobs shape: {logprobs.shape}")

        print(f"[Rank {rank}] Completed {num_iterations} forward passes")

        # Check for recompilation warnings
        for warning in w:
            msg = str(warning.message)
            if "recompile_limit" in msg or "AsyncCollectiveTensor" in msg:
                recompile_warnings.append(msg)
                print(f"[Rank {rank}] Warning: {msg[:200]}...")

    # Report results
    if recompile_warnings:
        print(
            f"[Rank {rank}] FAILED: Got {len(recompile_warnings)} "
            f"recompilation warnings"
        )
        # Cleanup before failing
        engine.destroy()
        if dist.is_initialized():
            dist.destroy_process_group()
        raise AssertionError(
            f"Got {len(recompile_warnings)} recompilation warnings. "
            f"First warning: {recompile_warnings[0][:200]}..."
        )

    # Verify output shape
    assert logprobs.shape[0] == batch_size, (
        f"Expected batch_size={batch_size}, got {logprobs.shape[0]}"
    )

    # Verify outputs are valid (not NaN/Inf)
    assert not torch.isnan(logprobs).any(), "Output contains NaN"
    assert not torch.isinf(logprobs).any(), "Output contains Inf"

    print(f"[Rank {rank}] PASSED: No recompilation warnings with TP + AC + compile")
    print(f"[Rank {rank}] Sample logprobs (first 5): {logprobs[0, :5]}")

    # Cleanup
    engine.destroy()

    if dist.is_initialized():
        dist.barrier()
        print(f"[Rank {rank}] All ranks completed successfully")
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Test TP + AC + compile compatibility")
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["qwen3"],
        default="qwen3",
        help="Type of model to test",
    )
    parser.add_argument(
        "--tp_size",
        type=int,
        default=2,
        help="Tensor Parallel size (must divide world_size)",
    )
    args = parser.parse_args()
    test_tp_ac_compile_compatibility(args.model_type, args.tp_size)


if __name__ == "__main__":
    main()
