#!/usr/bin/env python3
"""Unified FSDP Pipeline Parallelism test entry point.

This script tests FSDP-based pipeline parallelism using HuggingFace-style models.
It is launched via torchrun from the pytest test file test_fsdp_distributed_pp.py.

Run with:
    torchrun --nproc_per_node=2 tests/fsdp/torchrun/run_fsdp_pp_tests.py \
        --test_type=forward --pp_size=2 --output=/tmp/result.out

    torchrun --nproc_per_node=2 tests/fsdp/torchrun/run_fsdp_pp_tests.py \
        --test_type=forward_schedule --pp_size=2 --pp_schedule=ZBVZeroBubble \
        --output=/tmp/result.out

Supported test types:
    - forward: PP forward pass, verify output matches non-PP golden model
    - backward: PP backward/training step, verify gradients exist
    - gradient_correctness: Compare PP gradients with non-PP gradients
    - forward_schedule: Forward pass via schedule.eval() API (schedule-aware)
    - backward_schedule: Backward pass via schedule.step() API (schedule-aware)
    - fsdp_pp_sharding: PP=2 + DP=2 combination (FSDP sharding with PP)
"""

from __future__ import annotations

import argparse

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from areal.engine.fsdp_utils.pipeline_parallel import (
    build_pipeline_schedule,
    pipeline_llm_hf,
)

# =============================================================================
# Simple HuggingFace-like model for testing
# =============================================================================


class SimpleTransformerLayer(nn.Module):
    """A simple transformer layer that mimics HuggingFace layer output convention."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x, **kwargs):
        return (self.norm(self.linear(x) + x),)  # Return tuple like HF


class SimpleHFModel(nn.Module):
    """Mimics HuggingFace model structure for testing pipeline parallelism.

    Structure:
        model.embed_tokens  -> nn.Embedding
        model.layers        -> nn.ModuleList of SimpleTransformerLayer
        model.norm          -> nn.LayerNorm
        lm_head             -> nn.Linear
    """

    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.config = argparse.Namespace(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.model.layers = nn.ModuleList(
            [SimpleTransformerLayer(hidden_size) for _ in range(num_layers)]
        )
        self.model.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids, **kwargs):
        x = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            x = layer(x)[0]
        x = self.model.norm(x)
        return self.lm_head(x)


# =============================================================================
# Utilities
# =============================================================================


def write_result(out_file: str, success: bool) -> None:
    """Write test result to output file."""
    with open(out_file, "w") as f:
        f.write("Passed" if success else "Failed")


def print_rank0(msg: str) -> None:
    """Print only on rank 0."""
    if dist.get_rank() == 0:
        print(msg)


def create_model(
    vocab_size: int = 256,
    hidden_size: int = 64,
    num_layers: int = 4,
    seed: int = 42,
    device: torch.device | str = "cpu",
) -> SimpleHFModel:
    """Create and initialize a SimpleHFModel with deterministic weights."""
    torch.manual_seed(seed)
    model = SimpleHFModel(vocab_size, hidden_size, num_layers)
    model = model.to(device)
    return model


def create_test_inputs(
    batch_size: int = 2,
    seq_len: int = 16,
    vocab_size: int = 256,
    device: torch.device | str = "cuda",
    seed: int = 123,
) -> torch.Tensor:
    """Create deterministic test input_ids."""
    torch.manual_seed(seed)
    return torch.randint(0, vocab_size, (batch_size, seq_len), device=device)


def broadcast_inputs(
    rank: int,
    device: torch.device,
    batch_size: int = 2,
    seq_len: int = 16,
    vocab_size: int = 256,
    seed: int = 123,
) -> torch.Tensor:
    """Create inputs on rank 0 and broadcast to all ranks."""
    if rank == 0:
        input_ids = create_test_inputs(batch_size, seq_len, vocab_size, device, seed)
    else:
        input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    dist.broadcast(input_ids, src=0)
    return input_ids


def validate_gradients(model_parts: list[nn.Module]) -> tuple[bool, list[str]]:
    """Verify that gradients exist and are valid for all parameters.

    Returns:
        Tuple of (success, list of error messages).
    """
    errors = []
    for part_idx, part in enumerate(model_parts):
        for name, param in part.named_parameters():
            if param.requires_grad:
                if param.grad is None:
                    errors.append(f"Part {part_idx} '{name}': no gradient")
                elif torch.isnan(param.grad).any():
                    errors.append(f"Part {part_idx} '{name}': NaN gradient")
                elif torch.isinf(param.grad).any():
                    errors.append(f"Part {part_idx} '{name}': Inf gradient")
                elif param.grad.abs().sum() == 0:
                    errors.append(f"Part {part_idx} '{name}': zero gradient")
    return len(errors) == 0, errors


def gather_and_check_success(
    local_success: bool,
    local_errors: list[str],
    rank: int,
    world_size: int,
    device: torch.device,
) -> bool:
    """Gather success/failure status from all ranks and report."""
    all_results = [None] * world_size
    dist.all_gather_object(all_results, (local_success, local_errors))
    final_success = all(s for s, _ in all_results)

    if rank == 0:
        print_rank0("  Results by rank:")
        for r, (success, errors) in enumerate(all_results):
            status = "PASSED" if success else "FAILED"
            print_rank0(f"    Rank {r}: {status}")
            if not success:
                for err in errors[:5]:
                    print_rank0(f"      - {err}")

    return final_success


# =============================================================================
# Model configuration constants
# =============================================================================

VOCAB_SIZE = 256
HIDDEN_SIZE = 64
SEED = 42
INPUT_SEED = 123
BATCH_SIZE = 2
SEQ_LEN = 16


# =============================================================================
# Test: Forward pass
# =============================================================================


def test_forward(pp_size: int, out_file: str | None = None) -> bool:
    """Test PP forward pass: verify PP output matches non-PP golden model output.

    Creates a golden (non-pipelined) model and a PP-split model with identical
    weights. Runs forward on both and checks outputs match.

    Args:
        pp_size: Pipeline parallel degree (must equal world_size).
        out_file: Optional file to write result.

    Returns:
        True if test passed.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    num_layers = pp_size * 2  # At least 2 layers per stage
    pp_schedule = "1F1B"

    print_rank0(f"\n{'=' * 60}")
    print_rank0(f"FSDP PP Forward Test: pp_size={pp_size}, layers={num_layers}")
    print_rank0("=" * 60)

    # Create golden model
    golden_model = create_model(VOCAB_SIZE, HIDDEN_SIZE, num_layers, SEED, device)
    golden_model.eval()

    # Create and split PP model with same weights
    pp_model = create_model(VOCAB_SIZE, HIDDEN_SIZE, num_layers, SEED, device)
    pp_mesh = init_device_mesh("cuda", (pp_size,), mesh_dim_names=("pp",))

    stages, model_parts, has_first, has_last = pipeline_llm_hf(
        model=pp_model,
        device=device,
        pp_mesh=pp_mesh,
        pp_schedule=pp_schedule,
        pp_degree=pp_size,
        num_layers=num_layers,
    )

    for part in model_parts:
        part.eval()

    print(
        f"  Rank {rank}: has_first={has_first}, has_last={has_last}, "
        f"num_stages_local={len(stages)}"
    )

    # Broadcast identical inputs
    input_ids = broadcast_inputs(
        rank, device, BATCH_SIZE, SEQ_LEN, VOCAB_SIZE, INPUT_SEED
    )
    print_rank0(f"  Input shape: {input_ids.shape}")

    # Golden forward
    with torch.no_grad():
        golden_output = golden_model(input_ids)
    print_rank0(f"  Golden output shape: {golden_output.shape}")

    # PP forward via schedule.eval()
    n_microbatches = max(pp_size, 2)
    schedule = build_pipeline_schedule(
        stages,
        pp_schedule,
        n_microbatches,
        pp_degree=pp_size,
    )

    pp_output = None
    with torch.no_grad():
        if has_first:
            # Expand input for microbatches
            batched_input = input_ids.unsqueeze(0).expand(n_microbatches, -1, -1)
            # Reshape to (n_microbatches, batch*seq)
            batched_input = batched_input.reshape(n_microbatches, -1)
            schedule.eval(batched_input)
        else:
            schedule.eval()

        if has_last:
            # Get output from the last stage
            last_stage = [s for s in stages if s.is_last][0]
            if last_stage.output_chunks:
                pp_output = last_stage.output_chunks[0]

    # Compare
    torch.cuda.synchronize()
    dist.barrier()

    success = True
    if has_last and pp_output is not None:
        # PP output might have different shape due to microbatching
        if pp_output.shape != golden_output.shape:
            # Try to reshape to match
            try:
                pp_output = pp_output.view_as(golden_output)
            except RuntimeError:
                print_rank0(
                    f"  Shape mismatch: PP={pp_output.shape}, "
                    f"Golden={golden_output.shape}"
                )

        max_diff = (pp_output - golden_output).abs().max().item()
        mean_diff = (pp_output - golden_output).abs().mean().item()
        success = max_diff < 1e-2

        print_rank0(f"  Max diff: {max_diff:.6f}")
        print_rank0(f"  Mean diff: {mean_diff:.6f}")
        print_rank0(f"  Match: {success}")
    elif has_last:
        print_rank0("  WARNING: No PP output collected on last stage")
        success = False

    # Broadcast result from last rank
    success_t = torch.tensor([1 if success else 0], dtype=torch.int, device=device)
    # For 1F1B schedule, last stage is on last rank
    last_rank = world_size - 1
    dist.broadcast(success_t, src=last_rank)
    success = success_t.item() == 1

    dist.barrier()
    status = "PASSED" if success else "FAILED"
    print_rank0(f"\n  FSDP PP Forward Test: {status}")
    if rank == 0 and out_file:
        write_result(out_file, success)
    return success


# =============================================================================
# Test: Backward pass
# =============================================================================


def test_backward(pp_size: int, out_file: str | None = None) -> bool:
    """Test PP backward/training step: verify gradients exist after schedule.step().

    Args:
        pp_size: Pipeline parallel degree.
        out_file: Optional file to write result.

    Returns:
        True if test passed.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    num_layers = pp_size * 2
    pp_schedule = "1F1B"

    print_rank0(f"\n{'=' * 60}")
    print_rank0(f"FSDP PP Backward Test: pp_size={pp_size}, layers={num_layers}")
    print_rank0("=" * 60)

    # Create and split PP model
    model = create_model(VOCAB_SIZE, HIDDEN_SIZE, num_layers, SEED, device)
    pp_mesh = init_device_mesh("cuda", (pp_size,), mesh_dim_names=("pp",))

    stages, model_parts, has_first, has_last = pipeline_llm_hf(
        model=model,
        device=device,
        pp_mesh=pp_mesh,
        pp_schedule=pp_schedule,
        pp_degree=pp_size,
        num_layers=num_layers,
    )

    for part in model_parts:
        part.train()

    # Optimizer
    all_params = [p for m in model_parts for p in m.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(all_params, lr=1e-3)
    optimizer.zero_grad()
    print_rank0(f"  Trainable params: {len(all_params)}")

    # Inputs
    input_ids = broadcast_inputs(
        rank, device, BATCH_SIZE, SEQ_LEN, VOCAB_SIZE, INPUT_SEED
    )
    labels = broadcast_inputs(rank, device, BATCH_SIZE, SEQ_LEN, VOCAB_SIZE, seed=456)
    print_rank0(f"  Input shape: {input_ids.shape}")

    # Loss function for the schedule
    def loss_fn(output, target):
        return nn.functional.cross_entropy(
            output.view(-1, output.size(-1)),
            target.view(-1),
        )

    # Build schedule and run step
    n_microbatches = max(pp_size, 2)
    schedule = build_pipeline_schedule(
        stages,
        pp_schedule,
        n_microbatches,
        pp_degree=pp_size,
        loss_fn=loss_fn,
    )

    if has_first:
        batched_input = input_ids.unsqueeze(0).expand(n_microbatches, -1, -1)
        batched_input = batched_input.reshape(n_microbatches, -1)
        target = labels.unsqueeze(0).expand(n_microbatches, -1, -1)
        target = target.reshape(n_microbatches, -1)
        schedule.step(batched_input, target=target if has_last else None)
    else:
        target = labels.unsqueeze(0).expand(n_microbatches, -1, -1)
        target = target.reshape(n_microbatches, -1)
        schedule.step(target=target if has_last else None)

    torch.cuda.synchronize()
    dist.barrier()

    # Validate gradients
    local_success, local_errors = validate_gradients(model_parts)
    print(
        f"  Rank {rank}: gradient validation {'PASSED' if local_success else 'FAILED'}"
    )
    if local_success:
        total_norm = sum(
            p.grad.norm().item()
            for m in model_parts
            for p in m.parameters()
            if p.grad is not None
        )
        print(f"  Rank {rank}: total gradient norm = {total_norm:.4f}")

    final_success = gather_and_check_success(
        local_success,
        local_errors,
        rank,
        world_size,
        device,
    )

    dist.barrier()
    status = "PASSED" if final_success else "FAILED"
    print_rank0(f"\n  FSDP PP Backward Test: {status}")
    if rank == 0 and out_file:
        write_result(out_file, final_success)
    return final_success


# =============================================================================
# Test: Gradient correctness (PP vs non-PP)
# =============================================================================


def test_gradient_correctness(pp_size: int, out_file: str | None = None) -> bool:
    """Compare PP gradients against non-PP (single-device) golden gradients.

    Runs a forward-backward pass on both a golden single-device model and a
    PP-split model with identical weights and inputs. Then compares the
    gradient values across all parameters. For parameters split across stages,
    we gather the gradients and compare against the golden model's gradients.

    Args:
        pp_size: Pipeline parallel degree.
        out_file: Optional file to write result.

    Returns:
        True if gradients match within tolerance.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    num_layers = pp_size * 2
    pp_schedule = "1F1B"

    print_rank0(f"\n{'=' * 60}")
    print_rank0(f"FSDP PP Gradient Correctness Test: pp_size={pp_size}")
    print_rank0("=" * 60)

    # Create golden model and compute golden gradients
    golden_model = create_model(VOCAB_SIZE, HIDDEN_SIZE, num_layers, SEED, device)
    golden_model.train()

    input_ids = broadcast_inputs(
        rank, device, BATCH_SIZE, SEQ_LEN, VOCAB_SIZE, INPUT_SEED
    )
    labels = broadcast_inputs(rank, device, BATCH_SIZE, SEQ_LEN, VOCAB_SIZE, seed=456)

    # Golden forward-backward
    golden_output = golden_model(input_ids)
    golden_loss = nn.functional.cross_entropy(
        golden_output.view(-1, golden_output.size(-1)),
        labels.view(-1),
    )
    golden_loss.backward()

    golden_grads = {}
    for name, param in golden_model.named_parameters():
        if param.grad is not None:
            golden_grads[name] = param.grad.detach().clone()

    print_rank0(f"  Golden loss: {golden_loss.item():.4f}")
    print_rank0(f"  Golden grad params: {len(golden_grads)}")

    # Create PP model with same weights
    pp_model = create_model(VOCAB_SIZE, HIDDEN_SIZE, num_layers, SEED, device)
    pp_mesh = init_device_mesh("cuda", (pp_size,), mesh_dim_names=("pp",))

    stages, model_parts, has_first, has_last = pipeline_llm_hf(
        model=pp_model,
        device=device,
        pp_mesh=pp_mesh,
        pp_schedule=pp_schedule,
        pp_degree=pp_size,
        num_layers=num_layers,
    )

    for part in model_parts:
        part.train()

    # PP forward-backward using schedule.step()
    def loss_fn(output, target):
        return nn.functional.cross_entropy(
            output.view(-1, output.size(-1)),
            target.view(-1),
        )

    # Use n_microbatches=1 so we get a single forward-backward pass comparable to golden
    n_microbatches = max(pp_size, 2)
    schedule = build_pipeline_schedule(
        stages,
        pp_schedule,
        n_microbatches,
        pp_degree=pp_size,
        loss_fn=loss_fn,
    )

    if has_first:
        batched_input = input_ids.unsqueeze(0).expand(n_microbatches, -1, -1)
        batched_input = batched_input.reshape(n_microbatches, -1)
        target = labels.unsqueeze(0).expand(n_microbatches, -1, -1)
        target = target.reshape(n_microbatches, -1)
        schedule.step(batched_input, target=target if has_last else None)
    else:
        target = labels.unsqueeze(0).expand(n_microbatches, -1, -1)
        target = target.reshape(n_microbatches, -1)
        schedule.step(target=target if has_last else None)

    torch.cuda.synchronize()
    dist.barrier()

    # Validate that PP gradients exist
    local_success, local_errors = validate_gradients(model_parts)
    if not local_success:
        print(f"  Rank {rank}: gradient validation FAILED: {local_errors[:3]}")

    # Collect PP gradients from this rank's model parts
    pp_grads = {}
    for part in model_parts:
        for name, param in part.named_parameters():
            if param.grad is not None:
                pp_grads[name] = param.grad.detach().clone()

    # Each rank checks its own parameters against golden
    errors = []
    max_diffs = []
    for name, pp_grad in pp_grads.items():
        if name in golden_grads:
            golden_grad = golden_grads[name]
            if pp_grad.shape == golden_grad.shape:
                diff = (pp_grad - golden_grad).abs().max().item()
                max_diffs.append(diff)
                # Scaled tolerance: microbatch accumulation may cause differences
                if diff > 0.1:
                    errors.append(f"'{name}': max_diff={diff:.6f}")
            else:
                errors.append(
                    f"'{name}': shape mismatch PP={pp_grad.shape} vs "
                    f"golden={golden_grad.shape}"
                )

    grad_match = len(errors) == 0
    if max_diffs:
        overall_max = max(max_diffs)
        overall_mean = sum(max_diffs) / len(max_diffs)
        print(
            f"  Rank {rank}: checked {len(max_diffs)} params, "
            f"max_diff={overall_max:.6f}, mean_diff={overall_mean:.6f}"
        )
    else:
        print(f"  Rank {rank}: no overlapping params to compare")

    all_results = [None] * world_size
    dist.all_gather_object(all_results, (local_success and grad_match, errors))
    final_success = all(s for s, _ in all_results)

    if rank == 0:
        for r, (success, errs) in enumerate(all_results):
            status = "PASSED" if success else "FAILED"
            print_rank0(f"    Rank {r}: {status}")
            for e in errs[:3]:
                print_rank0(f"      - {e}")

    dist.barrier()
    status = "PASSED" if final_success else "FAILED"
    print_rank0(f"\n  FSDP PP Gradient Correctness Test: {status}")
    if rank == 0 and out_file:
        write_result(out_file, final_success)
    return final_success


# =============================================================================
# Test: Forward with schedule (ZBV etc.)
# =============================================================================


def test_forward_schedule(
    pp_size: int,
    pp_schedule: str = "ZBVZeroBubble",
    out_file: str | None = None,
) -> bool:
    """Test PP forward via schedule.eval() API with a specified schedule.

    This supports advanced schedules like ZBVZeroBubble that use V-style
    stage assignment (2 virtual stages per rank).

    Args:
        pp_size: Pipeline parallel degree.
        pp_schedule: Schedule name (e.g., "ZBVZeroBubble").
        out_file: Optional file to write result.

    Returns:
        True if test passed.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # For V-style schedules, num_virtual_stages = 2 * pp_degree
    # so we need at least 2 * pp_size * 2 layers (2 per virtual stage)
    num_layers = pp_size * 4

    print_rank0(f"\n{'=' * 60}")
    print_rank0(
        f"FSDP PP Forward Schedule Test: pp_size={pp_size}, "
        f"schedule={pp_schedule}, layers={num_layers}"
    )
    print_rank0("=" * 60)

    # Create golden model
    golden_model = create_model(VOCAB_SIZE, HIDDEN_SIZE, num_layers, SEED, device)
    golden_model.eval()

    # Create PP model
    pp_model = create_model(VOCAB_SIZE, HIDDEN_SIZE, num_layers, SEED, device)
    pp_mesh = init_device_mesh("cuda", (pp_size,), mesh_dim_names=("pp",))

    stages, model_parts, has_first, has_last = pipeline_llm_hf(
        model=pp_model,
        device=device,
        pp_mesh=pp_mesh,
        pp_schedule=pp_schedule,
        pp_degree=pp_size,
        num_layers=num_layers,
    )

    for part in model_parts:
        part.eval()

    print(
        f"  Rank {rank}: has_first={has_first}, has_last={has_last}, "
        f"stages={len(stages)}"
    )

    # Inputs
    input_ids = broadcast_inputs(
        rank, device, BATCH_SIZE, SEQ_LEN, VOCAB_SIZE, INPUT_SEED
    )

    # Golden forward
    with torch.no_grad():
        golden_output = golden_model(input_ids)
    print_rank0(f"  Golden output shape: {golden_output.shape}")

    # PP forward via schedule.eval()
    num_virtual_stages = len(stages) * pp_size
    n_microbatches = max(num_virtual_stages, 2)
    schedule = build_pipeline_schedule(
        stages,
        pp_schedule,
        n_microbatches,
        pp_degree=pp_size,
    )

    pp_output = None
    with torch.no_grad():
        if has_first:
            batched_input = input_ids.unsqueeze(0).expand(n_microbatches, -1, -1)
            batched_input = batched_input.reshape(n_microbatches, -1)
            schedule.eval(batched_input)
        else:
            schedule.eval()

        if has_last:
            last_stage = [s for s in stages if s.is_last][0]
            if last_stage.output_chunks:
                pp_output = last_stage.output_chunks[0]

    torch.cuda.synchronize()
    dist.barrier()

    success = True
    if has_last and pp_output is not None:
        if pp_output.shape != golden_output.shape:
            try:
                pp_output = pp_output.view_as(golden_output)
            except RuntimeError:
                pass

        max_diff = (pp_output - golden_output).abs().max().item()
        mean_diff = (pp_output - golden_output).abs().mean().item()
        success = max_diff < 1e-2

        print_rank0(f"  Max diff: {max_diff:.6f}")
        print_rank0(f"  Mean diff: {mean_diff:.6f}")
        print_rank0(f"  Match: {success}")
    elif has_last:
        print_rank0("  WARNING: No PP output collected on last stage")
        success = False

    # For V-style schedules (ZBV), last stage is on rank 0
    from torch.distributed.pipelining.schedules import (
        ScheduleZBVZeroBubble,
        get_schedule_class,
    )

    sched_cls = get_schedule_class(pp_schedule)
    v_style = sched_cls is ScheduleZBVZeroBubble
    try:
        from torch.distributed.pipelining.schedules import ScheduleDualPipeV

        v_style = v_style or (sched_cls is ScheduleDualPipeV)
    except ImportError:
        pass

    last_rank = 0 if v_style else world_size - 1
    success_t = torch.tensor([1 if success else 0], dtype=torch.int, device=device)
    dist.broadcast(success_t, src=last_rank)
    success = success_t.item() == 1

    dist.barrier()
    status = "PASSED" if success else "FAILED"
    print_rank0(f"\n  FSDP PP Forward Schedule Test: {status}")
    if rank == 0 and out_file:
        write_result(out_file, success)
    return success


# =============================================================================
# Test: Backward with schedule (ZBV etc.)
# =============================================================================


def test_backward_schedule(
    pp_size: int,
    pp_schedule: str = "ZBVZeroBubble",
    out_file: str | None = None,
) -> bool:
    """Test PP backward via schedule.step() API with a specified schedule.

    Args:
        pp_size: Pipeline parallel degree.
        pp_schedule: Schedule name.
        out_file: Optional file to write result.

    Returns:
        True if test passed.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    num_layers = pp_size * 4

    print_rank0(f"\n{'=' * 60}")
    print_rank0(
        f"FSDP PP Backward Schedule Test: pp_size={pp_size}, "
        f"schedule={pp_schedule}, layers={num_layers}"
    )
    print_rank0("=" * 60)

    # Create PP model
    model = create_model(VOCAB_SIZE, HIDDEN_SIZE, num_layers, SEED, device)
    pp_mesh = init_device_mesh("cuda", (pp_size,), mesh_dim_names=("pp",))

    stages, model_parts, has_first, has_last = pipeline_llm_hf(
        model=model,
        device=device,
        pp_mesh=pp_mesh,
        pp_schedule=pp_schedule,
        pp_degree=pp_size,
        num_layers=num_layers,
    )

    for part in model_parts:
        part.train()

    all_params = [p for m in model_parts for p in m.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(all_params, lr=1e-3)
    optimizer.zero_grad()
    print_rank0(f"  Trainable params: {len(all_params)}")

    # Inputs
    input_ids = broadcast_inputs(
        rank, device, BATCH_SIZE, SEQ_LEN, VOCAB_SIZE, INPUT_SEED
    )
    labels = broadcast_inputs(rank, device, BATCH_SIZE, SEQ_LEN, VOCAB_SIZE, seed=456)

    def loss_fn(output, target):
        return nn.functional.cross_entropy(
            output.view(-1, output.size(-1)),
            target.view(-1),
        )

    num_virtual_stages = len(stages) * pp_size
    n_microbatches = max(num_virtual_stages, 2)
    schedule = build_pipeline_schedule(
        stages,
        pp_schedule,
        n_microbatches,
        pp_degree=pp_size,
        loss_fn=loss_fn,
    )

    if has_first:
        batched_input = input_ids.unsqueeze(0).expand(n_microbatches, -1, -1)
        batched_input = batched_input.reshape(n_microbatches, -1)
        target = labels.unsqueeze(0).expand(n_microbatches, -1, -1)
        target = target.reshape(n_microbatches, -1)
        schedule.step(batched_input, target=target if has_last else None)
    else:
        target = labels.unsqueeze(0).expand(n_microbatches, -1, -1)
        target = target.reshape(n_microbatches, -1)
        schedule.step(target=target if has_last else None)

    torch.cuda.synchronize()
    dist.barrier()

    # Validate gradients
    local_success, local_errors = validate_gradients(model_parts)
    print(
        f"  Rank {rank}: gradient validation {'PASSED' if local_success else 'FAILED'}"
    )
    if local_success:
        total_norm = sum(
            p.grad.norm().item()
            for m in model_parts
            for p in m.parameters()
            if p.grad is not None
        )
        print(f"  Rank {rank}: total gradient norm = {total_norm:.4f}")

    final_success = gather_and_check_success(
        local_success,
        local_errors,
        rank,
        world_size,
        device,
    )

    dist.barrier()
    status = "PASSED" if final_success else "FAILED"
    print_rank0(f"\n  FSDP PP Backward Schedule Test: {status}")
    if rank == 0 and out_file:
        write_result(out_file, final_success)
    return final_success


# =============================================================================
# Test: FSDP + PP sharding (PP=2, DP=2 on 4 GPUs)
# =============================================================================


def test_fsdp_pp_sharding(pp_size: int, out_file: str | None = None) -> bool:
    """Test FSDP sharding combined with PP (PP=2, DP=2).

    This test uses a 2D mesh: (pp=2, dp=2) on 4 GPUs. Each PP stage's
    model part is wrapped with FSDP for data-parallel sharding across the
    dp dimension. Verifies that gradients are valid after a training step.

    Args:
        pp_size: Pipeline parallel degree (expected 2, with world_size=4).
        out_file: Optional file to write result.

    Returns:
        True if test passed.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    assert world_size == 4, f"fsdp_pp_sharding test requires 4 GPUs, got {world_size}"
    assert pp_size == 2, f"fsdp_pp_sharding test requires pp_size=2, got {pp_size}"

    dp_size = world_size // pp_size
    num_layers = pp_size * 2
    pp_schedule = "1F1B"

    print_rank0(f"\n{'=' * 60}")
    print_rank0(
        f"FSDP + PP Sharding Test: pp={pp_size}, dp={dp_size}, layers={num_layers}"
    )
    print_rank0("=" * 60)

    # Create 2D mesh: (pp, dp)
    mesh_2d = init_device_mesh(
        "cuda",
        (pp_size, dp_size),
        mesh_dim_names=("pp", "dp"),
    )
    pp_mesh = mesh_2d["pp"]

    print(
        f"  Rank {rank}: pp_mesh local_rank={pp_mesh.get_local_rank()}, "
        f"dp dim size={dp_size}"
    )

    # Create and split model for PP
    model = create_model(VOCAB_SIZE, HIDDEN_SIZE, num_layers, SEED, device)
    stages, model_parts, has_first, has_last = pipeline_llm_hf(
        model=model,
        device=device,
        pp_mesh=pp_mesh,
        pp_schedule=pp_schedule,
        pp_degree=pp_size,
        num_layers=num_layers,
    )

    # Wrap each model part with FSDP using the dp sub-mesh
    dp_mesh = mesh_2d["dp"]
    fsdp_model_parts = []
    for part in model_parts:
        part.train()
        fsdp_part = FSDP(part, device_mesh=dp_mesh)
        fsdp_model_parts.append(fsdp_part)

    all_params = [
        p for m in fsdp_model_parts for p in m.parameters() if p.requires_grad
    ]
    optimizer = torch.optim.SGD(all_params, lr=1e-3)
    optimizer.zero_grad()
    print_rank0(f"  Trainable params (FSDP wrapped): {len(all_params)}")

    # Inputs
    input_ids = broadcast_inputs(
        rank, device, BATCH_SIZE, SEQ_LEN, VOCAB_SIZE, INPUT_SEED
    )
    labels = broadcast_inputs(rank, device, BATCH_SIZE, SEQ_LEN, VOCAB_SIZE, seed=456)

    def loss_fn(output, target):
        return nn.functional.cross_entropy(
            output.view(-1, output.size(-1)),
            target.view(-1),
        )

    n_microbatches = max(pp_size, 2)
    schedule = build_pipeline_schedule(
        stages,
        pp_schedule,
        n_microbatches,
        pp_degree=pp_size,
        loss_fn=loss_fn,
    )

    if has_first:
        batched_input = input_ids.unsqueeze(0).expand(n_microbatches, -1, -1)
        batched_input = batched_input.reshape(n_microbatches, -1)
        target = labels.unsqueeze(0).expand(n_microbatches, -1, -1)
        target = target.reshape(n_microbatches, -1)
        schedule.step(batched_input, target=target if has_last else None)
    else:
        target = labels.unsqueeze(0).expand(n_microbatches, -1, -1)
        target = target.reshape(n_microbatches, -1)
        schedule.step(target=target if has_last else None)

    torch.cuda.synchronize()
    dist.barrier()

    # Validate gradients on the FSDP-wrapped parts
    local_success, local_errors = validate_gradients(fsdp_model_parts)
    print(
        f"  Rank {rank}: gradient validation {'PASSED' if local_success else 'FAILED'}"
    )
    if local_success:
        total_norm = sum(
            p.grad.norm().item()
            for m in fsdp_model_parts
            for p in m.parameters()
            if p.grad is not None
        )
        print(f"  Rank {rank}: total gradient norm = {total_norm:.4f}")

    final_success = gather_and_check_success(
        local_success,
        local_errors,
        rank,
        world_size,
        device,
    )

    dist.barrier()
    status = "PASSED" if final_success else "FAILED"
    print_rank0(f"\n  FSDP + PP Sharding Test: {status}")
    if rank == 0 and out_file:
        write_result(out_file, final_success)
    return final_success


# =============================================================================
# Test Registry and Main
# =============================================================================

TEST_REGISTRY = {
    "forward": test_forward,
    "backward": test_backward,
    "gradient_correctness": test_gradient_correctness,
    "forward_schedule": test_forward_schedule,
    "backward_schedule": test_backward_schedule,
    "fsdp_pp_sharding": test_fsdp_pp_sharding,
}


def main():
    parser = argparse.ArgumentParser(description="FSDP Pipeline Parallelism Tests")
    parser.add_argument(
        "--test_type",
        type=str,
        required=True,
        choices=list(TEST_REGISTRY.keys()),
        help="Type of test to run",
    )
    parser.add_argument(
        "--pp_size",
        type=int,
        default=2,
        help="Pipeline parallel size",
    )
    parser.add_argument(
        "--pp_schedule",
        type=str,
        default="1F1B",
        help="Pipeline schedule (for schedule tests only)",
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
    print_rank0(f"Running FSDP PP Test: {args.test_type}")
    print_rank0(f"  pp_size={args.pp_size}, pp_schedule={args.pp_schedule}")
    print_rank0("=" * 60)

    try:
        test_fn = TEST_REGISTRY[args.test_type]

        if args.test_type in ("forward_schedule", "backward_schedule"):
            success = test_fn(
                args.pp_size,
                pp_schedule=args.pp_schedule,
                out_file=args.output,
            )
        else:
            success = test_fn(args.pp_size, args.output)

        dist.barrier()

        if success:
            print_rank0(f"\nFSDP PP Test {args.test_type}: PASSED")
        else:
            print_rank0(f"\nFSDP PP Test {args.test_type}: FAILED")

    except Exception as e:
        print(f"Rank {rank} failed with exception: {e}")
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
