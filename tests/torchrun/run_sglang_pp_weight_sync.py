"""Distributed test worker for sglang PP weight synchronization.

Run via torchrun from test_sglang_pp_distributed.py.

Supports all three training engine types:
  - megatron: MegatronEngine (uses megatron.core parallel_state)
  - fsdp:     FSDPEngine (single-rank FSDP, dist.get_rank() for head detection)
  - archon:   ArchonEngine (uses WeightSyncState for group management)

Test types:
  - group_init:  Verify per-PP-rank group names, head detection, world sizes.
  - weight_sync: Verify PP ranks have different parameter names (layer splits).
"""
import argparse
import logging
import os
import sys

import torch
import torch.distributed as dist

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
from areal.infra.platforms import current_platform

logger = logging.getLogger(__name__)

MODEL_PATH = get_model_path(
    "/storage/openpsi/models/Qwen__Qwen3-0.6B/", "Qwen/Qwen3-0.6B"
)


# ---------------------------------------------------------------------------
# Result writer
# ---------------------------------------------------------------------------

def _write_result(output_path: str, passed: bool) -> None:
    if output_path:
        with open(output_path, "w") as f:
            f.write("Passed" if passed else "Failed")


# ---------------------------------------------------------------------------
# Engine factory
# ---------------------------------------------------------------------------

def _make_engine(backend: str, mb_spec: MicroBatchSpec, engine_type: str):
    """Create and initialize the appropriate training engine.

    Args:
        backend: Allocation string, e.g. "megatron:d1p2t2".
        mb_spec: Micro-batch specification.
        engine_type: One of "megatron", "fsdp", "archon".

    Returns:
        The initialized engine instance.
    """
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

    if engine_type == "megatron":
        from areal.engine import MegatronEngine
        engine = MegatronEngine(config)
    elif engine_type == "fsdp":
        from areal.engine import FSDPEngine
        engine = FSDPEngine(config)
    elif engine_type == "archon":
        from areal.experimental.engine.archon_engine import ArchonEngine
        engine = ArchonEngine(config)
    else:
        raise ValueError(f"Unknown engine_type: {engine_type}")

    engine.create_process_group(parallel_strategy=alloc.parallel)
    engine.initialize(addr=None, ft_spec=ft_spec)
    return engine


# ---------------------------------------------------------------------------
# Engine-specific helpers for group name / head detection
# ---------------------------------------------------------------------------

def _get_weight_update_group_names(engine, engine_type: str) -> list[str]:
    """Return the list of weight update group names for the engine.

    Megatron: single string in engine.weight_update_group_name
    FSDP:     list in engine.weight_update_group_names
    Archon:   list in engine._weight_sync_state.group_names (or single
              engine._weight_sync_state.group_name)
    """
    if engine_type == "megatron":
        return [engine.weight_update_group_name]
    elif engine_type == "fsdp":
        return list(engine.weight_update_group_names)
    elif engine_type == "archon":
        state = getattr(engine, "_weight_sync_state", None)
        if state is None:
            return []
        if state.group_names:
            return list(state.group_names)
        return [state.group_name]
    return []


def _is_pp_head(engine, engine_type: str) -> bool:
    """Return whether this rank is a pipeline parallel head."""
    if engine_type == "megatron":
        return engine.is_pipeline_parallel_head()
    elif engine_type == "fsdp":
        # FSDP uses dist.get_rank() == 0 as the head.
        return dist.get_rank() == 0
    elif engine_type == "archon":
        return engine.is_pipeline_parallel_head()
    return False


def _get_pp_rank(engine, engine_type: str) -> int:
    """Return the pipeline parallel rank for this worker."""
    if engine_type == "megatron":
        from megatron.core import parallel_state as mpu
        return mpu.get_pipeline_model_parallel_rank()
    elif engine_type == "fsdp":
        # FSDP does not have PP partitioning within the training engine;
        # all FSDP ranks hold the full model. PP only applies on the
        # inference (sglang) side. Return 0 for consistency.
        return 0
    elif engine_type == "archon":
        return engine.pipeline_parallel_rank
    return 0


def _get_pp_size(engine, engine_type: str) -> int:
    """Return the pipeline parallel world size."""
    if engine_type == "megatron":
        from megatron.core import parallel_state as mpu
        return mpu.get_pipeline_model_parallel_world_size()
    elif engine_type == "fsdp":
        return 1  # FSDP training side has no PP partitioning
    elif engine_type == "archon":
        return engine.pipeline_parallel_world_size
    return 1


# ---------------------------------------------------------------------------
# Test: group_init
# ---------------------------------------------------------------------------

def test_group_init(backend: str, gen_pp_size: int, output: str, engine_type: str):
    """Verify per-PP-rank group names and head detection.

    Checks:
      1. The weight_update_group_name(s) match the expected per-PP-rank pattern.
      2. is_pipeline_parallel_head returns the correct value.
      3. For gen_pp_size > 1, per-PP world sizes are computed correctly.
    """
    rank = int(os.environ["RANK"])
    mb_spec = MicroBatchSpec(max_tokens_per_mb=256)
    engine = _make_engine(backend, mb_spec, engine_type)

    try:
        pp_rank = _get_pp_rank(engine, engine_type)
        pp_size = _get_pp_size(engine, engine_type)
        is_head = _is_pp_head(engine, engine_type)
        group_names = _get_weight_update_group_names(engine, engine_type)

        logger.info(
            "rank=%d engine=%s pp_rank=%d pp_size=%d is_head=%s groups=%s",
            rank, engine_type, pp_rank, pp_size, is_head, group_names,
        )

        # Verify group name follows the per-PP-rank pattern.
        if engine_type == "megatron":
            expected_group = f"update_weight_group_{pp_rank}"
            assert engine.weight_update_group_name == expected_group, (
                f"rank {rank}: expected group '{expected_group}', "
                f"got '{engine.weight_update_group_name}'"
            )
        elif engine_type == "fsdp":
            # FSDP uses weight_update_group_names (list).
            # Before init_weight_update_from_distributed is called, the list
            # may be empty. Just verify the attribute type.
            assert isinstance(engine.weight_update_group_names, list), (
                f"rank {rank}: expected list, got {type(engine.weight_update_group_names)}"
            )
        elif engine_type == "archon":
            state = getattr(engine, "_weight_sync_state", None)
            if state is not None:
                expected_group = f"update_weight_group_{pp_rank}"
                assert state.group_name == expected_group, (
                    f"rank {rank}: expected '{expected_group}', got '{state.group_name}'"
                )

        # Verify head detection.
        if engine_type == "megatron":
            from megatron.core import parallel_state as mpu
            dp_rank = mpu.get_data_parallel_rank(with_context_parallel=True)
            tp_rank = mpu.get_tensor_model_parallel_rank()
            expected_head = (dp_rank == 0 and tp_rank == 0)
            assert is_head == expected_head, (
                f"rank {rank}: expected is_pp_head={expected_head}, got {is_head}"
            )
        elif engine_type == "fsdp":
            assert is_head == (dist.get_rank() == 0), (
                f"rank {rank}: FSDP head should be rank 0"
            )

        # Verify per-PP world size computation.
        if gen_pp_size > 1:
            gen_alloc = ModelAllocation.from_str(f"sglang:d1p{gen_pp_size}t2")
            per_pp_world = gen_alloc.parallel.world_size // gen_pp_size
            assert per_pp_world == 2, (
                f"Expected per-PP world size 2, got {per_pp_world}"
            )

        current_platform.synchronize()
        dist.barrier()

        if rank == 0 and output:
            _write_result(output, True)
    except Exception:
        if rank == 0 and output:
            _write_result(output, False)
        raise
    finally:
        engine.destroy()


# ---------------------------------------------------------------------------
# Test: weight_sync
# ---------------------------------------------------------------------------

def test_weight_sync(backend: str, gen_pp_size: int, output: str, engine_type: str):
    """Verify that PP ranks hold different parameter sets.

    For PP > 1, different PP ranks should have different layer indices in
    their named parameters. For PP = 1, this just verifies basic model init.
    """
    rank = int(os.environ["RANK"])
    mb_spec = MicroBatchSpec(max_tokens_per_mb=256)
    engine = _make_engine(backend, mb_spec, engine_type)

    try:
        pp_rank = _get_pp_rank(engine, engine_type)
        pp_size = _get_pp_size(engine, engine_type)

        # Collect parameter names.
        param_names = set()
        for name, _ in engine.model.named_parameters():
            param_names.add(name)

        logger.info(
            "rank=%d engine=%s pp_rank=%d pp_size=%d n_params=%d",
            rank, engine_type, pp_rank, pp_size, len(param_names),
        )

        # In PP mode with engines that partition the model (megatron, archon),
        # different PP ranks should have different parameter names.
        if pp_size > 1 and engine_type in ("megatron", "archon"):
            # Get the PP communication group from the appropriate engine API.
            if engine_type == "megatron":
                from megatron.core import parallel_state as mpu
                pp_group = mpu.get_pipeline_model_parallel_group()
            else:
                # Archon uses its own parallel_dims for group management.
                pp_group = engine.parallel_dims.get_group("pp")

            all_param_names = [None] * pp_size
            dist.all_gather_object(all_param_names, param_names, group=pp_group)

            if rank == 0:
                overlap = all_param_names[0] & all_param_names[1]
                union = all_param_names[0] | all_param_names[1]
                logger.info(
                    "PP rank 0 params: %d, PP rank 1 params: %d, "
                    "overlap: %d, union: %d",
                    len(all_param_names[0]),
                    len(all_param_names[1]),
                    len(overlap),
                    len(union),
                )
                # PP ranks should have some non-overlapping parameters.
                # Embeddings and final norm may overlap, but layer params differ.
                assert len(union) > len(all_param_names[0]), (
                    "PP ranks should have different parameters but union "
                    "equals PP rank 0 set."
                )

        current_platform.synchronize()
        dist.barrier()

        if rank == 0 and output:
            _write_result(output, True)
    except Exception:
        if rank == 0 and output:
            _write_result(output, False)
        raise
    finally:
        engine.destroy()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Distributed test worker for sglang PP weight sync."
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="megatron:d1p2t2",
        help="Training engine allocation string.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write Passed/Failed result.",
    )
    parser.add_argument(
        "--test_type",
        type=str,
        choices=["group_init", "weight_sync"],
        default="group_init",
        help="Which test to run.",
    )
    parser.add_argument(
        "--gen_pp_size",
        type=int,
        default=1,
        help="Inference-side PP size for validation.",
    )
    parser.add_argument(
        "--engine_type",
        type=str,
        choices=["megatron", "fsdp", "archon"],
        default="megatron",
        help="Training engine type to instantiate.",
    )
    args = parser.parse_args()

    if args.test_type == "group_init":
        test_group_init(args.backend, args.gen_pp_size, args.output, args.engine_type)
    elif args.test_type == "weight_sync":
        test_weight_sync(args.backend, args.gen_pp_size, args.output, args.engine_type)


if __name__ == "__main__":
    main()
