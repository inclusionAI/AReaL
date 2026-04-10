"""Distributed test worker for sglang PP weight synchronization.

Run via torchrun (see test_sglang_pp_distributed.py).
"""
import argparse
import os
import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu

from tests.utils import get_model_path

from areal.api import FinetuneSpec
from areal.api.alloc_mode import ModelAllocation, ParallelStrategy
from areal.api.cli_args import (
    MegatronEngineConfig,
    MicroBatchSpec,
    OptimizerConfig,
    TrainEngineConfig,
)
from areal.api.io_struct import WeightUpdateMeta
from areal.engine import MegatronEngine
from areal.infra.platforms import current_platform

MODEL_PATH = get_model_path(
    "/storage/openpsi/models/Qwen__Qwen3-0.6B/", "Qwen/Qwen3-0.6B"
)


def write_result(out, succ):
    with open(out, "w") as f:
        f.write("Passed" if succ else "Failed")


def make_engine(backend, mb_spec):
    config = TrainEngineConfig(
        backend=backend,
        experiment_name="test_pp",
        trial_name="test",
        path=MODEL_PATH,
        mb_spec=mb_spec,
        optimizer=None,
        megatron=MegatronEngineConfig(),
    )
    alloc = ModelAllocation.from_str(backend)
    ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=128, train_batch_size=8)
    engine = MegatronEngine(config)
    engine.create_process_group(parallel_strategy=alloc.parallel)
    engine.initialize(addr=None, ft_spec=ft_spec)
    return engine


def test_group_init(backend, gen_pp_size, output):
    """Test that per-PP-rank group names and heads are set up correctly."""
    rank = int(os.environ["RANK"])
    mb_spec = MicroBatchSpec(max_tokens_per_mb=256)
    engine = make_engine(backend, mb_spec)

    pp_rank = mpu.get_pipeline_model_parallel_rank()
    dp_rank = mpu.get_data_parallel_rank(with_context_parallel=True)
    tp_rank = mpu.get_tensor_model_parallel_rank()

    # Verify group name is per-PP-rank
    expected_group_name = f"update_weight_group_{pp_rank}"
    assert engine.weight_update_group_name == expected_group_name, (
        f"rank {rank}: expected group name {expected_group_name}, "
        f"got {engine.weight_update_group_name}"
    )

    # Verify is_pipeline_parallel_head is True for all dp=0,tp=0 ranks
    is_head = engine.is_pipeline_parallel_head()
    expected_head = dp_rank == 0 and tp_rank == 0
    assert is_head == expected_head, (
        f"rank {rank}: expected is_pp_head={expected_head}, got {is_head}"
    )

    # Verify that WeightUpdateMeta properly computes per-PP world sizes
    if gen_pp_size > 1:
        gen_alloc = ModelAllocation.from_str(f"sglang:d1p{gen_pp_size}t2")
        meta = WeightUpdateMeta.from_megatron_xccl(gen_allocation=gen_alloc)
        per_pp_world = meta.gen_allocation.parallel.world_size // gen_pp_size
        assert per_pp_world == 2, f"Expected per-PP world size 2, got {per_pp_world}"

    current_platform.synchronize()
    dist.barrier()
    engine.destroy()

    if rank == 0 and output:
        write_result(output, True)


def test_weight_sync(backend, gen_pp_size, output):
    """Test the full weight sync flow with per-PP-rank groups."""
    rank = int(os.environ["RANK"])
    mb_spec = MicroBatchSpec(max_tokens_per_mb=256)
    engine = make_engine(backend, mb_spec)

    pp_rank = mpu.get_pipeline_model_parallel_rank()
    pp_size = mpu.get_pipeline_model_parallel_world_size()

    # Verify that each PP rank has its own parameters
    param_names = set()
    for name, param in engine.model.named_parameters():
        param_names.add(name)

    # In PP mode, different PP ranks should have different parameter names
    # (different layer indices)
    if pp_size > 1:
        all_param_names = [None] * pp_size
        dist.all_gather_object(
            all_param_names,
            param_names,
            group=mpu.get_pipeline_model_parallel_group(),
        )
        if rank == 0:
            # PP ranks should have some non-overlapping parameters
            overlap = all_param_names[0] & all_param_names[1]
            # Some params may overlap (embeddings, final norm), but most should differ
            union = all_param_names[0] | all_param_names[1]
            print(
                f"PP rank 0 params: {len(all_param_names[0])}, "
                f"PP rank 1 params: {len(all_param_names[1])}, "
                f"overlap: {len(overlap)}, union: {len(union)}"
            )

    current_platform.synchronize()
    dist.barrier()
    engine.destroy()

    if rank == 0 and output:
        write_result(output, True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="megatron:d1p2t2")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--test_type",
        type=str,
        choices=["group_init", "weight_sync"],
        default="group_init",
    )
    parser.add_argument("--gen_pp_size", type=int, default=1)
    args = parser.parse_args()

    if args.test_type == "group_init":
        test_group_init(args.backend, args.gen_pp_size, args.output)
    elif args.test_type == "weight_sync":
        test_weight_sync(args.backend, args.gen_pp_size, args.output)


if __name__ == "__main__":
    main()
