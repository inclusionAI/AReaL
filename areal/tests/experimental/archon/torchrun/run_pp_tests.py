#!/usr/bin/env python3
"""Unified PP test entry point for low-level PP tests.

This script consolidates PP forward and backward tests into a single file,
following the pattern of run_ep_tests.py.

Run with:
    torchrun --nproc_per_node=2 areal/tests/experimental/archon/torchrun/run_pp_tests.py \
        --test_type=forward --pp_size=2 --output=/tmp/result.out

    torchrun --nproc_per_node=4 areal/tests/experimental/archon/torchrun/run_pp_tests.py \
        --test_type=backward --pp_size=4 --output=/tmp/result.out

Supported test types:
    - forward: Test PP forward pass matches golden model output
    - backward: Test PP backward pass - verify gradients flow correctly
"""

from __future__ import annotations

import argparse

import torch
import torch.distributed as dist
import torch.nn as nn

from areal.experimental.models.archon import ArchonParallelDims
from areal.experimental.models.archon.pipeline_parallel import (
    generate_llm_fqn_per_model_part,
    pipeline_module_split,
)
from areal.experimental.models.archon.qwen3 import Qwen3Model
from areal.tests.experimental.archon.torchrun.dist_utils import (
    create_dense_model_args,
    create_golden_model,
    create_test_input,
    print_rank0,
    verify_outputs_match,
    write_result,
)

# =============================================================================
# PP Forward Test
# =============================================================================


def test_pp_forward(pp_size: int, out_file: str | None = None) -> bool:
    """Test PP forward pass matches golden model output.

    This test verifies that a model split across multiple PP stages produces
    the same output as the non-split golden model. It manually passes activations
    between stages using point-to-point communication.

    Args:
        pp_size: Pipeline parallel degree (must equal world_size for this test)
        out_file: Optional file to write result ("Passed"/"Failed")

    Returns:
        True if test passed, False otherwise
    """
    # 1. Initialize distributed
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    print_rank0(f"\n{'=' * 60}")
    print_rank0(f"PP Forward Test: pp_size={pp_size}, world_size={world_size}")
    print_rank0("=" * 60)

    assert world_size == pp_size, (
        f"This test requires world_size == pp_size. "
        f"Got world_size={world_size}, pp_size={pp_size}"
    )

    # 2. Create model args (need enough layers for PP split)
    # Use n_layers = pp_size * 2 to ensure at least 2 layers per stage
    n_layers = pp_size * 2
    model_args = create_dense_model_args(
        n_layers=n_layers,
        dim=64,
        hidden_dim=128,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        vocab_size=1000,
    )

    # 3. Create golden model (non-PP) for comparison
    # Use same seed for reproducibility
    seed = 42
    golden_model = create_golden_model(model_args, device, seed=seed)
    golden_model.eval()

    # 4. Create PP parallel dims
    parallel_dims = ArchonParallelDims(
        pp=pp_size,
        dp_shard=1,
        tp=1,
        cp=1,
        world_size=world_size,
        device_type="cuda",
    )

    print_rank0(f"  PP enabled: {parallel_dims.pp_enabled}")
    print_rank0(f"  PP degree: {parallel_dims.pp}")

    # 5. Create PP model on meta device for efficient splitting
    # The model is created on meta device (no memory allocation), split into stages,
    # then each stage is materialized with weights on the target device
    with torch.device("meta"):
        model = Qwen3Model(model_args)

    # For debugging: create model parts without FSDP
    # This tests if PP forward works without FSDP interference
    pp_mesh = parallel_dims.get_mesh("pp")
    module_names_per_stage = generate_llm_fqn_per_model_part(
        num_stages=pp_size,
        num_layers=n_layers,
    )
    print_rank0(f"  Module names per stage: {module_names_per_stage}")

    pp_stages, model_parts = pipeline_module_split(
        whole_model=model,
        pp_mesh=pp_mesh,
        device=device,
        module_names_per_stage=module_names_per_stage,
    )

    # Materialize weights on each model part after splitting
    # Copy weights from golden model to ensure identical initialization
    golden_state = golden_model.state_dict()
    for part in model_parts:
        part.to_empty(device=device)
        part.init_buffers(buffer_device=device)
        # Load matching weights from golden model
        part_state = part.state_dict()
        for key in part_state:
            if key in golden_state:
                part_state[key].copy_(golden_state[key])
        part.load_state_dict(part_state)

    # Get PP local rank (which stage this rank belongs to)
    pp_local_rank = pp_mesh.get_local_rank()

    # Determine first/last stage
    has_first = any(s.is_first for s in pp_stages)
    has_last = any(s.is_last for s in pp_stages)

    print(
        f"  Rank {rank}: pp_local_rank={pp_local_rank}, has_first={has_first}, has_last={has_last}"
    )

    # Get PP group for P2P communication
    pp_group = parallel_dims.get_group("pp")
    pp_group_ranks = dist.get_process_group_ranks(pp_group)

    # Set to eval mode
    for part in model_parts:
        part.eval()

    # 6. Create test input (broadcast from rank 0 for consistency)
    if rank == 0:
        tokens, positions, cu_seqlens, max_seqlen = create_test_input(
            num_seqs=2,
            seq_len_per_seq=8,
            vocab_size=model_args.vocab_size,
            device=device,
            seed=123,
        )
        input_data = [tokens, positions, cu_seqlens, max_seqlen]
    else:
        input_data = [None, None, None, None]

    dist.broadcast_object_list(input_data, src=0)
    tokens, positions, cu_seqlens, max_seqlen = input_data

    # Move tensors to device
    tokens = tokens.to(device)
    positions = positions.to(device)
    cu_seqlens = cu_seqlens.to(device)

    print_rank0(f"  Input tokens shape: {tokens.shape}")

    # 7. Run golden model forward
    with torch.no_grad():
        golden_output = golden_model(tokens, positions, cu_seqlens, max_seqlen)

    print_rank0(f"  Golden output shape: {golden_output.shape}")

    # 8. Run PP forward manually by passing activations between stages
    # For PP with N stages, we sequentially:
    #   Stage 0: process tokens -> h0
    #   Stage 1: receive h0, process -> h1
    #   ...
    #   Stage N-1: receive h_{N-2}, process -> output
    print_rank0(f"\n  Starting PP forward pass through {pp_size} stages...")

    success = True
    max_diff = 0.0
    mean_diff = 0.0
    pp_output = None

    with torch.no_grad():
        # Initialize activation buffer
        # First stage uses tokens, others use hidden activations
        h = torch.zeros(
            (1, tokens.shape[1], model_args.dim),
            dtype=torch.float32,
            device=device,
        )

        # Sequential forward through all stages
        for stage_idx in range(pp_size):
            src_rank = pp_group_ranks[stage_idx]
            is_my_stage = pp_local_rank == stage_idx

            if stage_idx == 0:
                # First stage: process input tokens
                if is_my_stage:
                    print(f"  Rank {rank}: Stage {stage_idx} - processing input tokens")
                    h = model_parts[0](tokens, positions, cu_seqlens, max_seqlen)
                    print(f"  Rank {rank}: Stage {stage_idx} - output shape: {h.shape}")
            else:
                # Non-first stages: receive activation from previous stage, then process
                if is_my_stage:
                    print(
                        f"  Rank {rank}: Stage {stage_idx} - processing received activation"
                    )
                    h = model_parts[0](h, positions, cu_seqlens, max_seqlen)
                    print(f"  Rank {rank}: Stage {stage_idx} - output shape: {h.shape}")

            # Synchronize and broadcast activation to next stage (or to all for final)
            torch.cuda.synchronize()
            dist.barrier(group=pp_group)

            # Broadcast current stage's output to all ranks
            # This ensures the next stage has the activation
            if stage_idx < pp_size - 1:
                dist.broadcast(h, src=src_rank, group=pp_group)
                print_rank0(
                    f"  Broadcast activation from stage {stage_idx} (rank {src_rank})"
                )

        # Final output is on the last stage
        if has_last:
            pp_output = h

    # Synchronize after forward
    torch.cuda.synchronize()
    dist.barrier()

    # 9. Compare outputs (only on last stage)
    if has_last:
        success, max_diff, mean_diff = verify_outputs_match(
            pp_output, golden_output, rtol=1e-4, atol=1e-4, max_diff_threshold=1e-2
        )
        print_rank0("\n  Comparison results:")
        print_rank0(f"    PP output shape: {pp_output.shape}")
        print_rank0(f"    Golden output shape: {golden_output.shape}")
        print_rank0(f"    Max diff: {max_diff:.6f}")
        print_rank0(f"    Mean diff: {mean_diff:.6f}")
        print_rank0(f"    Outputs match: {success}")

        if not success:
            # Print more debug info on failure
            print_rank0(
                f"    PP output stats: min={pp_output.min():.4f}, max={pp_output.max():.4f}, mean={pp_output.mean():.4f}"
            )
            print_rank0(
                f"    Golden stats: min={golden_output.min():.4f}, max={golden_output.max():.4f}, mean={golden_output.mean():.4f}"
            )

    # 10. Broadcast result to all ranks (use last stage's global rank)
    last_stage_global_rank = pp_group_ranks[-1]
    success_tensor = torch.tensor([1 if success else 0], dtype=torch.int, device=device)
    dist.broadcast(success_tensor, src=last_stage_global_rank)
    success = success_tensor.item() == 1

    dist.barrier()

    if success:
        print_rank0(f"\n{'=' * 60}")
        print_rank0("PP Forward Test: PASSED")
        print_rank0("=" * 60)
    else:
        print_rank0(f"\n{'=' * 60}")
        print_rank0("PP Forward Test: FAILED")
        print_rank0("=" * 60)

    if rank == 0 and out_file:
        write_result(out_file, success)

    return success


# =============================================================================
# PP Backward Test
# =============================================================================


def validate_gradients_pp(model_parts: list[nn.Module]) -> tuple[bool, list[str]]:
    """Verify gradients flow through PP model parts.

    Args:
        model_parts: List of model parts from pipeline_module_split

    Returns:
        Tuple of (success, error_messages)
    """
    errors = []

    for part_idx, part in enumerate(model_parts):
        for name, param in part.named_parameters():
            if param.requires_grad:
                if param.grad is None:
                    errors.append(f"Part {part_idx} {name}: no gradient")
                elif torch.isnan(param.grad).any():
                    errors.append(f"Part {part_idx} {name}: NaN gradient")
                elif torch.isinf(param.grad).any():
                    errors.append(f"Part {part_idx} {name}: Inf gradient")
                elif param.grad.abs().sum() == 0:
                    errors.append(f"Part {part_idx} {name}: zero gradient")

    return len(errors) == 0, errors


def test_pp_backward(pp_size: int, out_file: str | None = None) -> bool:
    """Test PP backward pass - verify gradients flow through all stages.

    This test verifies that gradients flow correctly through a pipeline-parallel
    model. It manually passes activations forward and gradients backward between
    stages using point-to-point communication.

    Args:
        pp_size: Pipeline parallel degree (must equal world_size for this test)
        out_file: Optional file to write result ("Passed"/"Failed")

    Returns:
        True if test passed, False otherwise
    """
    # 1. Initialize distributed
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    print_rank0(f"\n{'=' * 60}")
    print_rank0(f"PP Backward Test: pp_size={pp_size}, world_size={world_size}")
    print_rank0("=" * 60)

    assert world_size == pp_size, (
        f"This test requires world_size == pp_size. "
        f"Got world_size={world_size}, pp_size={pp_size}"
    )

    # 2. Create model args (need enough layers for PP split)
    n_layers = pp_size * 2
    model_args = create_dense_model_args(
        n_layers=n_layers,
        dim=64,
        hidden_dim=128,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        vocab_size=1000,
    )

    # 3. Create PP parallel dims
    parallel_dims = ArchonParallelDims(
        pp=pp_size,
        dp_shard=1,
        tp=1,
        cp=1,
        world_size=world_size,
        device_type="cuda",
    )

    print_rank0(f"  PP enabled: {parallel_dims.pp_enabled}")
    print_rank0(f"  PP degree: {parallel_dims.pp}")

    # 4. Create PP model on meta device for efficient splitting
    with torch.device("meta"):
        model = Qwen3Model(model_args)

    # 5. Use pipeline_module_split (same as forward test, no FSDP)
    pp_mesh = parallel_dims.get_mesh("pp")
    module_names_per_stage = generate_llm_fqn_per_model_part(
        num_stages=pp_size,
        num_layers=n_layers,
    )
    print_rank0(f"  Module names per stage: {module_names_per_stage}")

    pp_stages, model_parts = pipeline_module_split(
        whole_model=model,
        pp_mesh=pp_mesh,
        device=device,
        module_names_per_stage=module_names_per_stage,
    )

    # Materialize weights on each model part after splitting
    # For backward test, we don't need to match golden model - just need valid weights
    seed = 42
    torch.manual_seed(seed)
    for part in model_parts:
        part.to_empty(device=device)
        part.init_weights()
        part.init_buffers(buffer_device=device)

    # Get PP local rank (which stage this rank belongs to)
    pp_local_rank = pp_mesh.get_local_rank()

    # Determine first/last stage
    has_first = any(s.is_first for s in pp_stages)
    has_last = any(s.is_last for s in pp_stages)

    print(
        f"  Rank {rank}: pp_local_rank={pp_local_rank}, has_first={has_first}, has_last={has_last}"
    )

    # Get PP group for communication
    pp_group = parallel_dims.get_group("pp")
    pp_group_ranks = dist.get_process_group_ranks(pp_group)

    # Set to train mode
    for part in model_parts:
        part.train()

    # 6. Create optimizer
    all_params = [p for m in model_parts for p in m.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(all_params, lr=1e-4)
    optimizer.zero_grad()

    print_rank0(f"  Trainable parameters: {len(all_params)}")

    # 7. Create test input and labels
    if rank == 0:
        tokens, positions, cu_seqlens, max_seqlen = create_test_input(
            num_seqs=2,
            seq_len_per_seq=8,
            vocab_size=model_args.vocab_size,
            device=device,
            seed=123,
        )
        # Create simple labels (shifted tokens for causal LM)
        labels = tokens.clone()
        input_data = [tokens, positions, cu_seqlens, max_seqlen, labels]
    else:
        input_data = [None, None, None, None, None]

    dist.broadcast_object_list(input_data, src=0)
    tokens, positions, cu_seqlens, max_seqlen, labels = input_data

    # Move tensors to device
    tokens = tokens.to(device)
    positions = positions.to(device)
    cu_seqlens = cu_seqlens.to(device)
    labels = labels.to(device)

    print_rank0(f"  Input tokens shape: {tokens.shape}")
    print_rank0(f"  Labels shape: {labels.shape}")

    # 8. Run PP forward + backward manually for multi-stage PP
    # For PP with N stages, we need to:
    # Forward: stage 0 -> stage 1 -> ... -> stage N-1
    # Backward: stage N-1 -> stage N-2 -> ... -> stage 0
    print_rank0(f"\n  Starting PP forward pass through {pp_size} stages...")

    # Store activations for backward pass
    # Each stage needs to remember its input activation for backward
    h_input = None  # Input to this stage (for backward)
    h_output = None  # Output from this stage
    output = None  # Final output (only on last stage)

    # Initialize activation buffer for communication
    h_buffer = torch.zeros(
        (1, tokens.shape[1], model_args.dim),
        dtype=torch.float32,
        device=device,
    )

    # Forward pass - sequential through all stages
    for stage_idx in range(pp_size):
        is_my_stage = pp_local_rank == stage_idx
        src_rank = pp_group_ranks[stage_idx]

        if stage_idx == 0:
            # First stage: process input tokens
            if is_my_stage:
                print(f"  Rank {rank}: Stage {stage_idx} - forward with input tokens")
                h_output = model_parts[0](tokens, positions, cu_seqlens, max_seqlen)
                h_output.retain_grad()
                h_buffer = h_output.detach().clone()
                print(
                    f"  Rank {rank}: Stage {stage_idx} - output shape: {h_output.shape}"
                )
        else:
            # Non-first stages: receive activation, then process
            if is_my_stage:
                # Clone the received buffer and enable gradients
                h_input = h_buffer.clone().requires_grad_(True)
                print(
                    f"  Rank {rank}: Stage {stage_idx} - forward with received activation"
                )
                h_output = model_parts[0](h_input, positions, cu_seqlens, max_seqlen)
                h_output.retain_grad()
                h_buffer = h_output.detach().clone()
                print(
                    f"  Rank {rank}: Stage {stage_idx} - output shape: {h_output.shape}"
                )

                if pp_local_rank == pp_size - 1:
                    output = h_output

        # Synchronize and broadcast to next stage
        torch.cuda.synchronize()
        dist.barrier(group=pp_group)

        if stage_idx < pp_size - 1:
            dist.broadcast(h_buffer, src=src_rank, group=pp_group)
            print_rank0(
                f"  Broadcast activation from stage {stage_idx} (rank {src_rank})"
            )

    # Compute loss and start backward (only on last stage)
    print_rank0("\n  Starting PP backward pass...")

    grad_buffer = torch.zeros(
        (1, tokens.shape[1], model_args.dim),
        dtype=torch.float32,
        device=device,
    )

    if pp_local_rank == pp_size - 1:
        print(f"  Rank {rank}: Computing loss on last stage")
        output_flat = output.view(-1, output.size(-1))
        labels_flat = labels.view(-1)
        loss = nn.functional.cross_entropy(output_flat, labels_flat)
        print(f"  Rank {rank}: Loss = {loss.item():.4f}")

        # Backward through last stage
        loss.backward()

        # Get gradient to send to previous stage
        if h_input is not None and h_input.grad is not None:
            grad_buffer = h_input.grad.detach().clone()
            print(
                f"  Rank {rank}: Gradient norm to send: {grad_buffer.norm().item():.4f}"
            )
        else:
            print(
                f"  Rank {rank}: WARNING - h_input.grad is None (this is expected for pp_size=1)"
            )

    # Backward pass - reverse sequential through all stages
    for stage_idx in range(pp_size - 1, 0, -1):
        src_rank = pp_group_ranks[stage_idx]

        torch.cuda.synchronize()
        dist.barrier(group=pp_group)
        dist.broadcast(grad_buffer, src=src_rank, group=pp_group)
        print_rank0(f"  Broadcast gradient from stage {stage_idx} (rank {src_rank})")

        # Process backward at the previous stage
        if pp_local_rank == stage_idx - 1:
            print(f"  Rank {rank}: Stage {pp_local_rank} - backward pass")
            print(
                f"  Rank {rank}: Received gradient norm: {grad_buffer.norm().item():.4f}"
            )

            # Backward through this stage
            h_output.backward(grad_buffer)

            # Get gradient to send to previous stage (if not first stage)
            if pp_local_rank > 0 and h_input is not None and h_input.grad is not None:
                grad_buffer = h_input.grad.detach().clone()
                print(
                    f"  Rank {rank}: Gradient norm to send: {grad_buffer.norm().item():.4f}"
                )

    # Synchronize after backward
    torch.cuda.synchronize()
    dist.barrier()

    # 9. Validate gradients on this rank
    local_success, local_errors = validate_gradients_pp(model_parts)

    print(
        f"  Rank {rank}: gradient validation {'PASSED' if local_success else 'FAILED'}"
    )
    if not local_success:
        for err in local_errors[:5]:  # Print first 5 errors
            print(f"    {err}")
    else:
        # Print some gradient stats for verification
        total_grad_norm = sum(
            p.grad.norm().item()
            for m in model_parts
            for p in m.parameters()
            if p.grad is not None
        )
        print(f"  Rank {rank}: Total gradient norm: {total_grad_norm:.4f}")

    # 10. Gather results from all ranks
    all_results = [None] * world_size
    dist.all_gather_object(all_results, (local_success, local_errors))

    # 11. Check all stages passed
    final_success = all(success for success, _ in all_results)

    if rank == 0:
        print_rank0("\n  Results by rank:")
        for r, (success, errors) in enumerate(all_results):
            status = "PASSED" if success else "FAILED"
            print_rank0(f"    Rank {r}: {status}")
            if not success:
                for err in errors[:3]:
                    print_rank0(f"      - {err}")

    dist.barrier()

    if final_success:
        print_rank0(f"\n{'=' * 60}")
        print_rank0("PP Backward Test: PASSED")
        print_rank0("=" * 60)
    else:
        print_rank0(f"\n{'=' * 60}")
        print_rank0("PP Backward Test: FAILED")
        print_rank0("=" * 60)

    if rank == 0 and out_file:
        write_result(out_file, final_success)

    return final_success


# =============================================================================
# Test Registry and Main
# =============================================================================

TEST_REGISTRY = {
    "forward": test_pp_forward,
    "backward": test_pp_backward,
}


def main():
    parser = argparse.ArgumentParser(description="Unified PP Tests")
    parser.add_argument(
        "--test_type",
        type=str,
        required=True,
        choices=list(TEST_REGISTRY.keys()),
        help="Type of test to run (forward or backward)",
    )
    parser.add_argument(
        "--pp_size",
        type=int,
        default=2,
        help="Pipeline parallel size (must equal world_size)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for test result",
    )
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    print_rank0("=" * 60)
    print_rank0(f"Running PP Test: {args.test_type}")
    print_rank0("=" * 60)

    try:
        test_fn = TEST_REGISTRY[args.test_type]
        success = test_fn(args.pp_size, args.output)

        dist.barrier()

        if success:
            print_rank0(f"\n{'=' * 60}")
            print_rank0(f"PP Test {args.test_type}: PASSED")
            print_rank0("=" * 60)
        else:
            print_rank0(f"\n{'=' * 60}")
            print_rank0(f"PP Test {args.test_type}: FAILED")
            print_rank0("=" * 60)

    except Exception as e:
        print(f"Rank {rank} failed with: {e}")
        import traceback

        traceback.print_exc()
        if rank == 0 and args.output:
            write_result(args.output, False)
        raise

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
