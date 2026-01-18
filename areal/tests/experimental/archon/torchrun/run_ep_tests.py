#!/usr/bin/env python3
"""Unified EP (Expert Parallel) test entry point.

This script consolidates all EP tests and supports different parallel configurations:
- EP+TP: ep=world_size, tp=world_size, cp=1 (2 GPU)
- EP Only: ep=world_size, tp=1, cp=1 (2 GPU)
- EP+CP: ep=2, tp=1, cp=2 (4 GPU)

Run with:
    torchrun --nproc_per_node=2 areal/tests/experimental/archon/torchrun/run_ep_tests.py \
        --test_type=ep_tp_forward --output=/tmp/result.out

Supported test types:
    - ep_tp_forward: EP+TP forward numerical correctness
    - ep_tp_weight_sync: EP+TP weight gather and roundtrip
    - ep_only_forward: EP only forward (tp=1)
    - ep_only_weight_sync: EP only weight sync (tp=1)
    - ep_cp_forward: EP+CP forward (4 GPU)
    - ep_cp_weight_sync: EP+CP weight sync (4 GPU)
    - state_dict_update: State dict correctness after optimizer step
"""

import argparse

import torch
import torch.distributed as dist

from areal.experimental.models.archon import ArchonParallelDims
from areal.experimental.models.archon.qwen3.infra.parallelize import parallelize_qwen3
from areal.experimental.models.archon.qwen3.model.model import Qwen3Model
from areal.experimental.models.archon.ulysses import (
    ulysses_gather_output,
    ulysses_slice_inputs,
)
from areal.tests.experimental.archon.torchrun.utils import (
    create_golden_model,
    create_moe_model_args,
    create_test_input,
    gather_full_state_dict,
    print_rank0,
    verify_outputs_match,
    write_result,
)

# =============================================================================
# Parallel Configuration Helpers
# =============================================================================


def get_parallel_dims_ep_tp(world_size: int) -> ArchonParallelDims:
    """Get parallel dims for EP+TP configuration (ep=ws, tp=ws)."""
    return ArchonParallelDims(
        dp_shard=1,
        tp=world_size,
        cp=1,
        ep=world_size,
        world_size=world_size,
        device_type="cuda",
    )


def get_parallel_dims_ep_only(world_size: int) -> ArchonParallelDims:
    """Get parallel dims for EP only configuration (ep=ws, tp=1).

    Note: dp_shard * tp * cp must equal world_size.
    For EP only with 2 GPUs: dp_shard=2, tp=1, cp=1, ep=2
    """
    return ArchonParallelDims(
        dp_shard=world_size,
        tp=1,
        cp=1,
        ep=world_size,
        world_size=world_size,
        device_type="cuda",
    )


def get_parallel_dims_ep_cp(world_size: int) -> ArchonParallelDims:
    """Get parallel dims for EP+CP configuration (ep=2, cp=2, 4 GPU).

    Note: dp_shard * tp * cp must equal world_size.
    For EP+CP with 4 GPUs: dp_shard=2, tp=1, cp=2, ep=4
    """
    assert world_size == 4, "EP+CP requires 4 GPUs"
    return ArchonParallelDims(
        dp_shard=2,
        tp=1,
        cp=2,
        ep=world_size,  # ep=4 borrows from dp_shard * cp
        world_size=world_size,
        device_type="cuda",
    )


def create_ep_model(
    model_args,
    parallel_dims: ArchonParallelDims,
    device: torch.device,
    seed: int = 42,
) -> Qwen3Model:
    """Create and parallelize a model with EP."""
    torch.manual_seed(seed)
    model = Qwen3Model(model_args)
    model.init_weights(device)
    model = model.to(device)

    parallelize_qwen3(
        model=model,
        parallel_dims=parallel_dims,
        param_dtype=torch.float32,
        reduce_dtype=torch.float32,
        loss_parallel=False,
        cpu_offload=False,
        reshard_after_forward=True,
        ac_config=None,
        enable_compile=False,
    )

    return model


# =============================================================================
# Forward Tests
# =============================================================================


def test_forward(
    parallel_dims: ArchonParallelDims,
    config_name: str,
    output: str | None = None,
    ep_model: Qwen3Model | None = None,
) -> tuple[bool, Qwen3Model]:
    """Test forward numerical correctness for given parallel config.

    Verify: EP model output vs non-EP golden model output.

    Args:
        parallel_dims: Pre-created ArchonParallelDims instance (reused across tests).
        config_name: Name of the test configuration for logging.
        output: Optional output file path for test result.
        ep_model: Optional pre-created EP model to reuse. If None, creates a new one.

    Returns:
        Tuple of (success, ep_model) where ep_model can be reused in subsequent tests.
    """
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")

    num_experts = 4
    model_args = create_moe_model_args(num_experts=num_experts)

    # Create golden model (non-parallelized)
    golden_model = create_golden_model(model_args, device, seed=42)

    # Create EP model if not provided (reusing parallel_dims instance)
    if ep_model is None:
        ep_model = create_ep_model(model_args, parallel_dims, device, seed=42)

    # Create test input (full sequence)
    tokens, positions, cu_seqlens, seq_len_per_seq = create_test_input(
        num_seqs=4,
        seq_len_per_seq=8,
        vocab_size=model_args.vocab_size,
        device=device,
        seed=123,
    )

    # Get CP info for input slicing and output gathering
    cp_group = parallel_dims.get_group("cp")
    cp_size = parallel_dims.cp
    cp_rank = parallel_dims.get_mesh("cp").get_local_rank() if cp_size > 1 else 0

    # Prepare inputs for EP model (slice if CP enabled)
    if cp_size > 1:
        # Slice inputs for this CP rank
        inputs_dict = {"input_ids": tokens, "position_ids": positions}
        # Create a dummy labels tensor for ulysses_slice_inputs
        labels = torch.zeros_like(tokens)
        sliced_inputs, _ = ulysses_slice_inputs(inputs_dict, labels, cp_rank, cp_size)
        tokens_ep = sliced_inputs["input_ids"]
        positions_ep = sliced_inputs["position_ids"]
        # cu_seqlens should NOT be sliced - Ulysses gathers full sequence in attention
        # so cu_seqlens refers to the full sequence positions
        cu_seqlens_ep = cu_seqlens
        seq_len_per_seq_ep = seq_len_per_seq
    else:
        tokens_ep = tokens
        positions_ep = positions
        cu_seqlens_ep = cu_seqlens
        seq_len_per_seq_ep = seq_len_per_seq

    # Forward pass
    with torch.no_grad():
        output_golden = golden_model(tokens, positions, cu_seqlens, seq_len_per_seq)
        output_ep = ep_model(tokens_ep, positions_ep, cu_seqlens_ep, seq_len_per_seq_ep)

    # Gather EP output if CP enabled
    # Model output has shape [1, seq_len/cp_size, dim], so use seq_dim=1
    if cp_size > 1:
        output_ep = ulysses_gather_output(output_ep, cp_group, seq_dim=1)

    # Verify outputs match
    success, max_diff, mean_diff = verify_outputs_match(output_golden, output_ep)

    print_rank0(f"  {config_name} forward:")
    print_rank0(f"    Output shape: {output_ep.shape}")
    print_rank0(f"    Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")

    if success:
        print_rank0(f"  {config_name}_forward: PASSED")
    else:
        print_rank0(f"  {config_name}_forward: FAILED (max_diff={max_diff})")

    dist.barrier()

    if rank == 0 and output:
        write_result(output, success)

    return success, ep_model


def test_output_consistency_across_ranks(
    parallel_dims: ArchonParallelDims,
    config_name: str,
    ep_model: Qwen3Model | None = None,
) -> bool:
    """Test that all ranks produce the same gathered output for the same input.

    For CP-enabled models, each CP rank processes different sequence slices,
    so we compare the gathered (full) outputs across ranks.

    Args:
        parallel_dims: Pre-created ArchonParallelDims instance (reused across tests).
        config_name: Name of the test configuration for logging.
        ep_model: Optional pre-created EP model to reuse. If None, creates a new one.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    num_experts = 4
    model_args = create_moe_model_args(num_experts=num_experts)

    # Reuse EP model if provided, otherwise create a new one
    if ep_model is None:
        ep_model = create_ep_model(model_args, parallel_dims, device, seed=42)
    model = ep_model

    # Create test input (full sequence)
    tokens, positions, cu_seqlens, seq_len_per_seq = create_test_input(
        num_seqs=4,
        seq_len_per_seq=8,
        vocab_size=model_args.vocab_size,
        device=device,
        seed=123,
    )

    # Get CP info for input slicing and output gathering
    cp_group = parallel_dims.get_group("cp")
    cp_size = parallel_dims.cp
    cp_rank = parallel_dims.get_mesh("cp").get_local_rank() if cp_size > 1 else 0

    # Prepare inputs for model (slice if CP enabled)
    if cp_size > 1:
        inputs_dict = {"input_ids": tokens, "position_ids": positions}
        labels = torch.zeros_like(tokens)
        sliced_inputs, _ = ulysses_slice_inputs(inputs_dict, labels, cp_rank, cp_size)
        tokens_model = sliced_inputs["input_ids"]
        positions_model = sliced_inputs["position_ids"]
        # cu_seqlens should NOT be sliced - Ulysses gathers full sequence in attention
        cu_seqlens_model = cu_seqlens
        seq_len_per_seq_model = seq_len_per_seq
    else:
        tokens_model = tokens
        positions_model = positions
        cu_seqlens_model = cu_seqlens
        seq_len_per_seq_model = seq_len_per_seq

    # Synchronize all ranks before forward pass to ensure model state is consistent
    dist.barrier()

    # Forward pass
    with torch.no_grad():
        output = model(
            tokens_model, positions_model, cu_seqlens_model, seq_len_per_seq_model
        )

    # Gather output if CP enabled
    # Model output has shape [1, seq_len/cp_size, dim], so use seq_dim=1
    if cp_size > 1:
        output = ulysses_gather_output(output, cp_group, seq_dim=1)

    # Gather outputs from all ranks
    output_list = [torch.zeros_like(output) for _ in range(world_size)]
    dist.all_gather(output_list, output)

    # All outputs should be identical (after gathering CP outputs)
    success = True
    for i in range(1, world_size):
        if not torch.allclose(output_list[0], output_list[i], rtol=1e-5, atol=1e-5):
            max_diff = (output_list[0] - output_list[i]).abs().max().item()
            print_rank0(
                f"  Output mismatch between rank 0 and rank {i}: max_diff={max_diff}"
            )
            success = False

    if success:
        print_rank0(f"  {config_name}_output_consistency: PASSED")
    else:
        print_rank0(f"  {config_name}_output_consistency: FAILED")

    dist.barrier()
    return success


# =============================================================================
# Weight Sync Tests
# =============================================================================


def test_weight_sync(
    parallel_dims: ArchonParallelDims,
    config_name: str,
    output: str | None = None,
) -> bool:
    """Test weight gather, roundtrip, and cross-rank consistency.

    Verify:
    1. Gathered weights match original (before parallelization)
    2. Gathered weights are consistent across all ranks
    3. Loaded weights produce same output as original model

    Args:
        parallel_dims: Pre-created ArchonParallelDims instance (reused across tests).
        config_name: Name of the test configuration for logging.
        output: Optional output file path for test result.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    num_experts = 4
    model_args = create_moe_model_args(num_experts=num_experts)

    # Create model and save original weights before parallelization
    torch.manual_seed(42)
    model = Qwen3Model(model_args)
    model.init_weights(device)
    model = model.to(device)
    original_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Apply parallelization (reusing parallel_dims instance)
    parallelize_qwen3(
        model=model,
        parallel_dims=parallel_dims,
        param_dtype=torch.float32,
        reduce_dtype=torch.float32,
        loss_parallel=False,
        cpu_offload=False,
        reshard_after_forward=True,
        ac_config=None,
        enable_compile=False,
    )

    # Gather sharded weights
    gathered_state = gather_full_state_dict(model)

    success = True

    # Test 1: Verify expert weights match original
    print_rank0(f"  {config_name} weight sync:")
    for name in original_state:
        if "experts" in name and name in gathered_state:
            original = original_state[name]
            gathered = gathered_state[name]

            if original.shape != gathered.shape:
                print_rank0(
                    f"    Shape mismatch for {name}: {original.shape} vs {gathered.shape}"
                )
                success = False
                continue

            if not torch.allclose(original, gathered, rtol=1e-5, atol=1e-5):
                max_diff = (original - gathered).abs().max().item()
                print_rank0(f"    Weight mismatch for {name}: max_diff={max_diff}")
                success = False

    # Test 2: Verify cross-rank consistency
    expert_weight_names = [
        name for name in gathered_state.keys() if "experts.w1" in name
    ]

    for name in expert_weight_names[:2]:
        local_weight = gathered_state[name]
        weight_list = [torch.zeros_like(local_weight) for _ in range(world_size)]
        dist.all_gather(weight_list, local_weight)

        for i in range(1, world_size):
            if not torch.allclose(weight_list[0], weight_list[i], rtol=1e-5, atol=1e-5):
                max_diff = (weight_list[0] - weight_list[i]).abs().max().item()
                print_rank0(
                    f"    Cross-rank mismatch for {name} (rank 0 vs {i}): {max_diff}"
                )
                success = False

    # Test 3: Verify roundtrip (load gathered weights to new model)
    torch.manual_seed(42)
    model_new = Qwen3Model(model_args)
    model_new.init_weights(device)
    model_new = model_new.to(device)

    # Load gathered state
    loadable_state = {}
    model_new_state = model_new.state_dict()
    for name, tensor in gathered_state.items():
        if name in model_new_state and tensor.shape == model_new_state[name].shape:
            loadable_state[name] = tensor
    model_new.load_state_dict(loadable_state, strict=False)

    # Compare forward outputs
    tokens, positions, cu_seqlens, seq_len_per_seq = create_test_input(
        num_seqs=4,
        seq_len_per_seq=8,
        vocab_size=model_args.vocab_size,
        device=device,
        seed=456,
    )

    # Create reference model
    golden_model = create_golden_model(model_args, device, seed=42)

    with torch.no_grad():
        output_new = model_new(tokens, positions, cu_seqlens, seq_len_per_seq)
        output_ref = golden_model(tokens, positions, cu_seqlens, seq_len_per_seq)

    if not torch.allclose(output_new, output_ref, rtol=1e-5, atol=1e-5):
        max_diff = (output_new - output_ref).abs().max().item()
        print_rank0(f"    Roundtrip forward mismatch: max_diff={max_diff}")
        success = False

    if success:
        print_rank0(f"  {config_name}_weight_sync: PASSED")
    else:
        print_rank0(f"  {config_name}_weight_sync: FAILED")

    dist.barrier()

    if rank == 0 and output:
        write_result(output, success)

    return success


# =============================================================================
# State Dict Update Test
# =============================================================================


def test_state_dict_update(output: str | None = None) -> bool:
    """Test state dict correctness after optimizer step.

    Verify:
    1. Create EP model + optimizer
    2. Forward + backward + optimizer.step()
    3. Gather EP weights to full state_dict
    4. Create non-EP model, load gathered state_dict
    5. Two models with same input should produce same output
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    num_experts = 4
    model_args = create_moe_model_args(num_experts=num_experts)

    # Create EP model
    parallel_dims = get_parallel_dims_ep_tp(world_size)
    model = create_ep_model(model_args, parallel_dims, device, seed=42)

    # Create optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Training step
    tokens, positions, cu_seqlens, seq_len_per_seq = create_test_input(
        num_seqs=4,
        seq_len_per_seq=8,
        vocab_size=model_args.vocab_size,
        device=device,
        seed=123,
    )

    model_output = model(tokens, positions, cu_seqlens, seq_len_per_seq)
    loss = model_output.sum()
    loss.backward()
    optimizer.step()

    # Gather updated weights
    gathered_state = gather_full_state_dict(model)

    # Create new non-EP model and load gathered weights
    torch.manual_seed(
        99
    )  # Different seed to ensure weights are loaded, not initialized
    model_new = Qwen3Model(model_args)
    model_new.init_weights(device)
    model_new = model_new.to(device)

    loadable_state = {}
    model_new_state = model_new.state_dict()
    for name, tensor in gathered_state.items():
        if name in model_new_state and tensor.shape == model_new_state[name].shape:
            loadable_state[name] = tensor
    model_new.load_state_dict(loadable_state, strict=False)

    # Forward with new input
    tokens2, positions2, cu_seqlens2, seq_len_per_seq2 = create_test_input(
        num_seqs=4,
        seq_len_per_seq=8,
        vocab_size=model_args.vocab_size,
        device=device,
        seed=789,
    )

    with torch.no_grad():
        # EP model forward
        output_ep = model(tokens2, positions2, cu_seqlens2, seq_len_per_seq2)
        # Non-EP model forward
        output_new = model_new(tokens2, positions2, cu_seqlens2, seq_len_per_seq2)

    # Verify outputs match
    success, max_diff, mean_diff = verify_outputs_match(
        output_ep, output_new, rtol=1e-4, atol=1e-4
    )

    print_rank0("  state_dict_update:")
    print_rank0(f"    Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")

    if success:
        print_rank0("  state_dict_update: PASSED")
    else:
        print_rank0(f"  state_dict_update: FAILED (max_diff={max_diff})")

    dist.barrier()

    if rank == 0 and output:
        write_result(output, success)

    return success


# =============================================================================
# Test Runners
# =============================================================================


def run_ep_tp_forward(output: str | None = None) -> bool:
    """Run EP+TP forward test."""
    print_rank0("\n=== EP+TP Forward Test ===")
    world_size = dist.get_world_size()
    parallel_dims = get_parallel_dims_ep_tp(world_size)
    success, ep_model = test_forward(parallel_dims, "ep_tp", output)
    if success:
        success = test_output_consistency_across_ranks(parallel_dims, "ep_tp", ep_model)
    return success


def run_ep_tp_weight_sync(output: str | None = None) -> bool:
    """Run EP+TP weight sync test."""
    print_rank0("\n=== EP+TP Weight Sync Test ===")
    world_size = dist.get_world_size()
    parallel_dims = get_parallel_dims_ep_tp(world_size)
    return test_weight_sync(parallel_dims, "ep_tp", output)


def run_ep_only_forward(output: str | None = None) -> bool:
    """Run EP only forward test."""
    print_rank0("\n=== EP Only Forward Test ===")
    world_size = dist.get_world_size()
    parallel_dims = get_parallel_dims_ep_only(world_size)
    success, ep_model = test_forward(parallel_dims, "ep_only", output)
    if success:
        success = test_output_consistency_across_ranks(
            parallel_dims, "ep_only", ep_model
        )
    return success


def run_ep_only_weight_sync(output: str | None = None) -> bool:
    """Run EP only weight sync test."""
    print_rank0("\n=== EP Only Weight Sync Test ===")
    world_size = dist.get_world_size()
    parallel_dims = get_parallel_dims_ep_only(world_size)
    return test_weight_sync(parallel_dims, "ep_only", output)


def run_ep_cp_forward(output: str | None = None) -> bool:
    """Run EP+CP forward test (requires 4 GPUs)."""
    print_rank0("\n=== EP+CP Forward Test ===")
    world_size = dist.get_world_size()
    parallel_dims = get_parallel_dims_ep_cp(world_size)
    success, ep_model = test_forward(parallel_dims, "ep_cp", output)
    if success:
        success = test_output_consistency_across_ranks(parallel_dims, "ep_cp", ep_model)
    return success


def run_ep_cp_weight_sync(output: str | None = None) -> bool:
    """Run EP+CP weight sync test (requires 4 GPUs)."""
    print_rank0("\n=== EP+CP Weight Sync Test ===")
    world_size = dist.get_world_size()
    parallel_dims = get_parallel_dims_ep_cp(world_size)
    return test_weight_sync(parallel_dims, "ep_cp", output)


def run_state_dict_update(output: str | None = None) -> bool:
    """Run state dict update test."""
    print_rank0("\n=== State Dict Update Test ===")
    return test_state_dict_update(output)


# =============================================================================
# Main
# =============================================================================

TEST_REGISTRY = {
    "ep_tp_forward": run_ep_tp_forward,
    "ep_tp_weight_sync": run_ep_tp_weight_sync,
    "ep_only_forward": run_ep_only_forward,
    "ep_only_weight_sync": run_ep_only_weight_sync,
    "ep_cp_forward": run_ep_cp_forward,
    "ep_cp_weight_sync": run_ep_cp_weight_sync,
    "state_dict_update": run_state_dict_update,
}


def main():
    parser = argparse.ArgumentParser(description="EP Tests")
    parser.add_argument(
        "--test_type",
        type=str,
        required=True,
        choices=list(TEST_REGISTRY.keys()),
        help="Type of test to run",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for test result (Passed/Failed)",
    )
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()

    torch.cuda.set_device(rank)

    print_rank0("=" * 60)
    print_rank0(f"Running EP Test: {args.test_type}")
    print_rank0("=" * 60)

    try:
        test_fn = TEST_REGISTRY[args.test_type]
        success = test_fn(args.output)

        dist.barrier()

        if success:
            print_rank0(f"\n{'=' * 60}")
            print_rank0(f"EP Test {args.test_type}: PASSED")
            print_rank0("=" * 60)
        else:
            print_rank0(f"\n{'=' * 60}")
            print_rank0(f"EP Test {args.test_type}: FAILED")
            print_rank0("=" * 60)
            if rank == 0 and args.output:
                write_result(args.output, False)

    except Exception as e:
        print(f"Rank {rank} failed with: {e}")
        import traceback

        traceback.print_exc()
        if rank == 0 and args.output:
            write_result(args.output, False)
        raise

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
