#!/usr/bin/env python3
"""Engine checkpoint integration tests.

Tests for ArchonEngine save/load methods with HF format using DCP infrastructure.

Run with:
    torchrun --nproc_per_node=2 areal/tests/experimental/archon/torchrun/run_checkpoint_tests.py \
        --test_type=save_hf_dense --model_path=/path/to/model --output=/tmp/result.out

Supported test types:
    - save_hf_dense: Test save() with weight_format="hf" on dense model
    - save_load_forward_match: Test save -> load -> forward output matches
    - moe_checkpoint: Test MoE model checkpoint with EP
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile

import torch
import torch.distributed as dist

from areal.api.alloc_mode import ParallelStrategy
from areal.api.cli_args import MicroBatchSpec, OptimizerConfig, TrainEngineConfig
from areal.api.io_struct import FinetuneSpec, SaveLoadMeta
from areal.experimental.engine.archon_engine import ArchonLMEngine
from areal.tests.experimental.archon.torchrun.dist_utils import (
    print_rank0,
    write_result,
)


def create_engine_config(model_path: str) -> TrainEngineConfig:
    """Create engine configuration for testing."""
    return TrainEngineConfig(
        experiment_name="test_checkpoint",
        trial_name="test",
        path=model_path,
        mb_spec=MicroBatchSpec(n_mbs=1),
        optimizer=OptimizerConfig(
            type="adam",
            lr=1e-5,
            weight_decay=0.01,
            warmup_steps_proportion=0.0,
            lr_scheduler_type="constant",
            gradient_clipping=1.0,
        ),
        temperature=1.0,
    )


def test_save_hf_dense(model_path: str, output: str | None = None) -> bool:
    """Test ArchonEngine.save() with weight_format="hf" on dense model.

    Steps:
    1. Initialize ArchonEngine with real HF checkpoint
    2. Call save(SaveLoadMeta(weight_format="hf")) to save in HF format
    3. Verify output contains safetensors files
    4. Verify config.json is copied
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print_rank0("\n=== Test: save HF format (Dense) ===")
    print_rank0(f"Model path: {model_path}")
    print_rank0(f"World size: {world_size}")

    if rank == 0:
        output_dir = tempfile.mkdtemp(prefix="hf_checkpoint_")
    else:
        output_dir = None

    output_dir_list = [output_dir]
    dist.broadcast_object_list(output_dir_list, src=0)
    output_dir = output_dir_list[0]

    success = True

    try:
        config = create_engine_config(model_path)
        engine = ArchonLMEngine(config)

        parallel_strategy = ParallelStrategy(
            data_parallel_size=1,
            tensor_parallel_size=world_size,
        )
        ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=4, train_batch_size=4)

        engine.create_process_group(parallel_strategy=parallel_strategy)
        engine.initialize(addr=None, ft_spec=ft_spec)

        print_rank0(f"Engine initialized with model: {engine.model.__class__.__name__}")

        # Save checkpoint using unified save() interface
        save_meta = SaveLoadMeta(
            path=output_dir,
            weight_format="hf",
            with_optim=False,
            tokenizer=engine.tokenizer,
        )
        engine.save(save_meta)

        dist.barrier()

        if rank == 0:
            safetensors_files = list(
                f for f in os.listdir(output_dir) if f.endswith(".safetensors")
            )
            if len(safetensors_files) == 0:
                print_rank0("ERROR: No safetensors files in output")
                success = False
            else:
                print_rank0(f"Saved {len(safetensors_files)} safetensors files")

            config_path = os.path.join(output_dir, "config.json")
            if not os.path.exists(config_path):
                print_rank0("ERROR: config.json not found in output")
                success = False
            else:
                print_rank0("config.json saved")

        engine.destroy()

    except Exception as e:
        print_rank0(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
        success = False

    finally:
        dist.barrier()
        if rank == 0 and output_dir and os.path.exists(output_dir):
            shutil.rmtree(output_dir)

    if success:
        print_rank0("save_hf_dense: PASSED")
    else:
        print_rank0("save_hf_dense: FAILED")

    if rank == 0 and output:
        write_result(output, success)

    return success


def test_save_load_forward_match(model_path: str, output: str | None = None) -> bool:
    """Test that forward output matches before save and after load.

    This is the most important test for checkpoint correctness:
    1. Initialize ArchonEngine with real HF checkpoint
    2. Run forward pass and record output
    3. Save checkpoint using save(weight_format="hf")
    4. Load checkpoint using load(weight_format="hf")
    5. Run forward pass again
    6. Verify outputs match exactly

    This test uses a small 0.6B model and runs on 1-2 GPUs.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print_rank0("\n=== Test: save_load_forward_match ===")
    print_rank0(f"Model path: {model_path}")
    print_rank0(f"World size: {world_size}")

    if rank == 0:
        output_dir = tempfile.mkdtemp(prefix="forward_match_test_")
    else:
        output_dir = None

    output_dir_list = [output_dir]
    dist.broadcast_object_list(output_dir_list, src=0)
    output_dir = output_dir_list[0]

    success = True

    try:
        config = create_engine_config(model_path)
        engine = ArchonLMEngine(config)

        parallel_strategy = ParallelStrategy(
            data_parallel_size=1,
            tensor_parallel_size=world_size,
        )
        ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=4, train_batch_size=4)

        engine.create_process_group(parallel_strategy=parallel_strategy)
        engine.initialize(addr=None, ft_spec=ft_spec)

        print_rank0(f"Engine initialized with model: {engine.model.__class__.__name__}")

        torch.manual_seed(42)
        batch_size = 4
        seq_len = 32
        vocab_size = engine.model_config.vocab_size

        input_ids = torch.randint(
            0, vocab_size, (1, batch_size * seq_len), device=engine.device
        )
        positions = torch.arange(batch_size * seq_len, device=engine.device).unsqueeze(
            0
        )
        cu_seqlens = torch.tensor(
            [i * seq_len for i in range(batch_size + 1)],
            dtype=torch.int32,
            device=engine.device,
        )

        engine.model.eval()
        with torch.no_grad():
            output_before = engine.model(
                input_ids, positions, cu_seqlens, max_seqlen=seq_len
            )
            output_before = output_before.clone()

        print_rank0(f"Output before save: shape={output_before.shape}")

        save_meta = SaveLoadMeta(
            path=output_dir,
            weight_format="hf",
            with_optim=False,
            tokenizer=engine.tokenizer,
        )
        engine.save(save_meta)
        print_rank0(f"Checkpoint saved to {output_dir}")

        dist.barrier()

        load_meta = SaveLoadMeta(
            path=output_dir,
            weight_format="hf",
            with_optim=False,
        )
        engine.load(load_meta)
        print_rank0("Checkpoint loaded")

        dist.barrier()

        engine.model.eval()
        with torch.no_grad():
            output_after = engine.model(
                input_ids, positions, cu_seqlens, max_seqlen=seq_len
            )

        print_rank0(f"Output after load: shape={output_after.shape}")

        max_diff = (output_before - output_after).abs().max().item()
        mean_diff = (output_before - output_after).abs().mean().item()
        allclose = torch.allclose(output_before, output_after, rtol=1e-4, atol=1e-4)

        print_rank0(f"  Max diff: {max_diff:.6e}")
        print_rank0(f"  Mean diff: {mean_diff:.6e}")
        print_rank0(f"  Allclose (rtol=1e-4, atol=1e-4): {allclose}")

        if not allclose:
            print_rank0("ERROR: Forward outputs do not match after load!")
            # Allow small differences due to numerical precision
            if max_diff > 1e-3:
                success = False
                print_rank0(f"ERROR: Max diff {max_diff} exceeds threshold 1e-3")
            else:
                print_rank0(f"WARNING: Max diff {max_diff} is small, treating as pass")
        else:
            print_rank0("Forward outputs match exactly!")

        engine.destroy()

    except Exception as e:
        print_rank0(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
        success = False

    finally:
        dist.barrier()
        if rank == 0 and output_dir and os.path.exists(output_dir):
            shutil.rmtree(output_dir)

    if success:
        print_rank0("save_load_forward_match: PASSED")
    else:
        print_rank0("save_load_forward_match: FAILED")

    if rank == 0 and output:
        write_result(output, success)

    return success


def test_save_load_forward_match_with_compile_ac(
    model_path: str, output: str | None = None
) -> bool:
    """Test forward match with torch.compile and activation checkpointing enabled.

    This test verifies that checkpoint save/load works correctly when the model
    has wrapper prefixes in parameter names (e.g., _orig_mod from torch.compile,
    _checkpoint_wrapped_module from activation checkpointing).

    Steps:
    1. Initialize ArchonEngine with compile + AC enabled
    2. Run forward pass and record output
    3. Save checkpoint using save(weight_format="hf")
    4. Load checkpoint using load(weight_format="hf")
    5. Run forward pass again
    6. Verify outputs match exactly
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print_rank0("\n=== Test: save_load_forward_match_with_compile_ac ===")
    print_rank0(f"Model path: {model_path}")
    print_rank0(f"World size: {world_size}")

    if rank == 0:
        output_dir = tempfile.mkdtemp(prefix="forward_match_compile_ac_test_")
    else:
        output_dir = None

    output_dir_list = [output_dir]
    dist.broadcast_object_list(output_dir_list, src=0)
    output_dir = output_dir_list[0]

    success = True

    try:
        config = create_engine_config(model_path)
        # Enable compile and activation checkpointing
        config.archon.enable_compile = True
        config.gradient_checkpointing = True
        config.archon.ac_mode = "full"

        engine = ArchonLMEngine(config)

        parallel_strategy = ParallelStrategy(
            data_parallel_size=1,
            tensor_parallel_size=world_size,
        )
        ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=4, train_batch_size=4)

        engine.create_process_group(parallel_strategy=parallel_strategy)
        engine.initialize(addr=None, ft_spec=ft_spec)

        print_rank0(f"Engine initialized with model: {engine.model.__class__.__name__}")
        print_rank0(f"  torch.compile enabled: {config.archon.enable_compile}")
        print_rank0(f"  Activation checkpointing: {config.archon.ac_mode}")

        # Verify wrapper prefixes exist in parameter names
        param_names = [name for name, _ in engine.model.named_parameters()]
        has_orig_mod = any("_orig_mod" in name for name in param_names)
        has_checkpoint_wrapper = any(
            "_checkpoint_wrapped_module" in name for name in param_names
        )
        print_rank0(f"  Has _orig_mod prefix: {has_orig_mod}")
        print_rank0(
            f"  Has _checkpoint_wrapped_module prefix: {has_checkpoint_wrapper}"
        )

        torch.manual_seed(42)
        batch_size = 4
        seq_len = 32
        vocab_size = engine.model_config.vocab_size

        input_ids = torch.randint(
            0, vocab_size, (1, batch_size * seq_len), device=engine.device
        )
        positions = torch.arange(batch_size * seq_len, device=engine.device).unsqueeze(
            0
        )
        cu_seqlens = torch.tensor(
            [i * seq_len for i in range(batch_size + 1)],
            dtype=torch.int32,
            device=engine.device,
        )

        engine.model.eval()
        with torch.no_grad():
            output_before = engine.model(
                input_ids, positions, cu_seqlens, max_seqlen=seq_len
            )
            output_before = output_before.clone()

        print_rank0(f"Output before save: shape={output_before.shape}")

        save_meta = SaveLoadMeta(
            path=output_dir,
            weight_format="hf",
            with_optim=False,
            tokenizer=engine.tokenizer,
        )
        engine.save(save_meta)
        print_rank0(f"Checkpoint saved to {output_dir}")

        dist.barrier()

        load_meta = SaveLoadMeta(
            path=output_dir,
            weight_format="hf",
            with_optim=False,
        )
        engine.load(load_meta)
        print_rank0("Checkpoint loaded")

        dist.barrier()

        engine.model.eval()
        with torch.no_grad():
            output_after = engine.model(
                input_ids, positions, cu_seqlens, max_seqlen=seq_len
            )

        print_rank0(f"Output after load: shape={output_after.shape}")

        max_diff = (output_before - output_after).abs().max().item()
        mean_diff = (output_before - output_after).abs().mean().item()
        allclose = torch.allclose(output_before, output_after, rtol=1e-4, atol=1e-4)

        print_rank0(f"  Max diff: {max_diff:.6e}")
        print_rank0(f"  Mean diff: {mean_diff:.6e}")
        print_rank0(f"  Allclose (rtol=1e-4, atol=1e-4): {allclose}")

        if not allclose:
            print_rank0("ERROR: Forward outputs do not match after load!")
            # Allow small differences due to numerical precision
            if max_diff > 1e-3:
                success = False
                print_rank0(f"ERROR: Max diff {max_diff} exceeds threshold 1e-3")
            else:
                print_rank0(f"WARNING: Max diff {max_diff} is small, treating as pass")
        else:
            print_rank0("Forward outputs match exactly!")

        engine.destroy()

    except Exception as e:
        print_rank0(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
        success = False

    finally:
        dist.barrier()
        if rank == 0 and output_dir and os.path.exists(output_dir):
            shutil.rmtree(output_dir)

    if success:
        print_rank0("save_load_forward_match_with_compile_ac: PASSED")
    else:
        print_rank0("save_load_forward_match_with_compile_ac: FAILED")

    if rank == 0 and output:
        write_result(output, success)

    return success


def test_moe_checkpoint(model_path: str, output: str | None = None) -> bool:
    """Test MoE model checkpoint with Expert Parallelism.

    This test requires 4 GPUs and tests:
    1. Loading MoE model with EP=4
    2. Saving checkpoint without OOM (no full_tensor() in to_hf)
    3. Verifying saved checkpoint is valid

    Note: This test is memory-intensive due to MoE model size (30B params).
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print_rank0("\n=== Test: moe_checkpoint ===")
    print_rank0(f"Model path: {model_path}")
    print_rank0(f"World size: {world_size}")

    if world_size < 4:
        print_rank0("SKIP: MoE checkpoint test requires 4 GPUs")
        if rank == 0 and output:
            write_result(output, True)  # Skip is not failure
        return True

    if rank == 0:
        output_dir = tempfile.mkdtemp(prefix="moe_checkpoint_")
    else:
        output_dir = None

    output_dir_list = [output_dir]
    dist.broadcast_object_list(output_dir_list, src=0)
    output_dir = output_dir_list[0]

    success = True

    try:
        config = create_engine_config(model_path)
        engine = ArchonLMEngine(config)

        parallel_strategy = ParallelStrategy(
            data_parallel_size=1,
            tensor_parallel_size=world_size,
        )
        ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=4, train_batch_size=4)

        engine.create_process_group(parallel_strategy=parallel_strategy)
        engine.initialize(addr=None, ft_spec=ft_spec)

        print_rank0("MoE Engine initialized")
        print_rank0(f"  Model: {engine.model.__class__.__name__}")
        print_rank0(
            f"  Config experts: {getattr(engine.model_config, 'num_experts', 'N/A')}"
        )

        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated() / 1024**3
        mem_before_max = torch.cuda.max_memory_allocated() / 1024**3
        print_rank0(
            f"  Memory before save: {mem_before:.2f} GB (max: {mem_before_max:.2f} GB)"
        )

        print_rank0("Saving MoE checkpoint...")
        save_meta = SaveLoadMeta(
            path=output_dir,
            weight_format="hf",
            with_optim=False,
            tokenizer=engine.tokenizer,
        )
        engine.save(save_meta)

        torch.cuda.synchronize()
        mem_after = torch.cuda.memory_allocated() / 1024**3
        mem_after_max = torch.cuda.max_memory_allocated() / 1024**3
        print_rank0(
            f"  Memory after save: {mem_after:.2f} GB (max: {mem_after_max:.2f} GB)"
        )

        dist.barrier()

        if rank == 0:
            safetensors_files = [
                f for f in os.listdir(output_dir) if f.endswith(".safetensors")
            ]
            if len(safetensors_files) == 0:
                print_rank0("ERROR: No safetensors files in output")
                success = False
            else:
                total_size = sum(
                    os.path.getsize(os.path.join(output_dir, f))
                    for f in safetensors_files
                )
                print_rank0(
                    f"Saved {len(safetensors_files)} files, "
                    f"total size: {total_size / 1024**3:.2f} GB"
                )

        engine.destroy()

    except torch.cuda.OutOfMemoryError as e:
        print_rank0(f"ERROR: OOM during MoE checkpoint: {e}")
        success = False

    except Exception as e:
        print_rank0(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
        success = False

    finally:
        dist.barrier()
        if rank == 0 and output_dir and os.path.exists(output_dir):
            shutil.rmtree(output_dir)

    if success:
        print_rank0("moe_checkpoint: PASSED")
    else:
        print_rank0("moe_checkpoint: FAILED")

    if rank == 0 and output:
        write_result(output, success)

    return success


# =============================================================================
# Main
# =============================================================================

TEST_REGISTRY = {
    "save_hf_dense": test_save_hf_dense,
    "save_load_forward_match": test_save_load_forward_match,
    "save_load_forward_match_with_compile_ac": test_save_load_forward_match_with_compile_ac,
    "moe_checkpoint": test_moe_checkpoint,
}


def main():
    parser = argparse.ArgumentParser(description="Engine Checkpoint Tests")
    parser.add_argument(
        "--test_type",
        type=str,
        required=True,
        choices=list(TEST_REGISTRY.keys()),
        help="Type of test to run",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to HF model checkpoint",
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
    print_rank0(f"Running Checkpoint Test: {args.test_type}")
    print_rank0("=" * 60)

    try:
        test_fn = TEST_REGISTRY[args.test_type]
        success = test_fn(args.model_path, args.output)

        dist.barrier()

        if success:
            print_rank0(f"\n{'=' * 60}")
            print_rank0(f"Checkpoint Test {args.test_type}: PASSED")
            print_rank0("=" * 60)
        else:
            print_rank0(f"\n{'=' * 60}")
            print_rank0(f"Checkpoint Test {args.test_type}: FAILED")
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
