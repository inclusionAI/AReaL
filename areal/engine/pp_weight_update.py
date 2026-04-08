"""
Per-PP-rank NCCL weight update for AReaL Megatron Engine.

Architecture:
    Megatron PP=1 (single stage, owns ALL params)
    SGLang PP=2 (two stages, each owns a subset of params)

    OLD: 1 NCCL group with world_size = pp*tp + 1, all params broadcast to all workers
         -> 50% bandwidth waste, AND rank collision causes deadlock when PP>1

    NEW: pp_size separate NCCL groups, each with world_size = tp_size + 1
         Group "areal-pp_0": Megatron(rank=0) + SGLang PP0-TP0(rank=1) + SGLang PP0-TP1(rank=2)
         Group "areal-pp_1": Megatron(rank=0) + SGLang PP1-TP0(rank=1) + SGLang PP1-TP1(rank=2)

    KEY FIX: SGLang's internal PP forwarding sends every HTTP request to ALL PP*TP
    schedulers. To prevent non-target PP workers from joining the wrong group, we
    patch the SGLang scheduler to filter by pp_rank (see sglang_pp_patches.py).

Design reference: slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py
"""

from __future__ import annotations

import os
from concurrent.futures import Future
from contextlib import nullcontext
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu
from torch import nn

from areal.api import ParamSpec, WeightUpdateMeta
from areal.engine.core.distributed import init_custom_process_group
from areal.engine.megatron_utils.megatron import (
    all_gather_param,
    convert_to_hf,
    get_named_parameters,
    remove_padding,
)
from areal.infra.platforms import current_platform
from areal.utils import logging
from areal.utils.constants import DIST_GROUP_DEFAULT_TIMEOUT
from areal.utils.network import find_free_ports, format_host_for_url, gethostip
from areal.utils.perf_tracer import trace_perf

if TYPE_CHECKING:
    from areal.engine.megatron_engine import MegatronEngine

logger = logging.getLogger("PPWeightUpdate")


# ---------------------------------------------------------------------------
#  Mixin / patch functions for MegatronEngine
# ---------------------------------------------------------------------------


def pp_init_weight_update_from_distributed(self: "MegatronEngine", meta: WeightUpdateMeta) -> None:
    """Initialize per-PP-rank NCCL groups for weight updates.

    Creates one NCCL group per SGLang PP rank. Each group contains:
        - Megatron actor (rank 0 in every group, acting as broadcaster)
        - SGLang workers for that PP rank (ranks 1..tp_size)

    IMPORTANT: This uses the EXISTING self.rollout_engine (RolloutCallback)
    which routes through the standard /callback/init_weights_group endpoint.
    The SGLang scheduler filter (installed by sglang_pp_patches.py) ensures
    only the correct PP rank's workers join each group.
    """
    assert meta.type == "xccl"
    assert meta.gen_allocation is not None

    gen_parallel = meta.gen_allocation.parallel
    sglang_pp_size = gen_parallel.pp_size
    sglang_tp_size = gen_parallel.tp_size

    logger.info(
        "[pp_init] Starting per-PP-rank NCCL group initialization: "
        "sglang_pp_size=%d, sglang_tp_size=%d, megatron_pp_rank=%d",
        sglang_pp_size, sglang_tp_size,
        mpu.get_pipeline_model_parallel_rank(),
    )

    # Master address for NCCL rendezvous
    meta.nccl_master_address = self.weight_update_master_addr = gethostip()

    # Allocate one port per PP group
    ports = find_free_ports(sglang_pp_size)
    self._pp_weight_update_master_ports = ports
    # Set the first port on meta for backward compatibility
    meta.nccl_master_port = ports[0]

    os.environ["TORCHELASTIC_USE_AGENT_STORE"] = str(False)

    if not self.is_pipeline_parallel_head():
        logger.info("[pp_init] Not pipeline parallel head, skipping group creation.")
        return

    self.engine_lock.acquire()

    # Dict: pp_rank -> NCCL group
    self._pp_weight_update_groups: dict[int, object] = {}
    # Dict: pp_rank -> group_name
    self._pp_weight_update_group_names: dict[int, str] = {}

    # For each PP rank, create a separate NCCL group
    # We process them SEQUENTIALLY because:
    # 1. SGLang's scheduler processes requests sequentially
    # 2. init_custom_process_group is blocking (NCCL rendezvous)
    # 3. Both Megatron and SGLang must join each group before proceeding to the next
    for pp_rank in range(sglang_pp_size):
        group_name = f"areal-pp_{pp_rank}"
        self._pp_weight_update_group_names[pp_rank] = group_name

        # Each group: 1 Megatron + tp_size SGLang workers
        group_world_size = sglang_tp_size + 1
        port = ports[pp_rank]
        init_method = f"tcp://{format_host_for_url(meta.nccl_master_address)}:{port}"

        logger.info(
            "[pp_init] Creating NCCL group '%s': world_size=%d, "
            "init_method=%s, megatron_rank=0",
            group_name, group_world_size, init_method,
        )

        # Build per-PP-rank meta for the rollout side
        pp_meta = _build_pp_meta(meta, pp_rank, port, group_name, sglang_tp_size)

        # Tell rollout side to init this PP group (non-blocking Future)
        # This goes through: RolloutCallback -> /callback/init_weights_group ->
        # RolloutController -> workers -> RemoteInfEngine -> HTTP to SGLang ->
        # SGLang scheduler (filtered by pp_rank via our patch)
        fut = self.rollout_engine.init_weights_update_group(pp_meta)

        # Megatron joins the group as rank 0 (blocks until all members rendezvous)
        logger.info("[pp_init] Megatron joining group '%s' as rank 0...", group_name)
        group = init_custom_process_group(
            backend=current_platform.communication_backend,
            world_size=group_world_size,
            init_method=init_method,
            rank=0,
            group_name=group_name,
            timeout=DIST_GROUP_DEFAULT_TIMEOUT,
        )
        self._pp_weight_update_groups[pp_rank] = group
        logger.info("[pp_init] Megatron joined group '%s' successfully.", group_name)

        # Wait for rollout side to finish
        logger.info("[pp_init] Waiting for rollout side of group '%s'...", group_name)
        fut.result()
        logger.info("[pp_init] Rollout side of group '%s' ready.", group_name)

    self.engine_lock.release()
    logger.info("[pp_init] All %d per-PP-rank groups initialized.", sglang_pp_size)


def _build_pp_meta(
    base_meta: WeightUpdateMeta,
    pp_rank: int,
    port: int,
    group_name: str,
    tp_size: int,
) -> WeightUpdateMeta:
    """Build a WeightUpdateMeta for a specific PP rank's NCCL group.

    CRITICAL: nccl_group_name is set to "areal-pp_{N}" which:
    1. SGLangBackend uses to build the HTTP request with correct group_name
    2. SGLang scheduler filter uses to determine target pp_rank
    3. ModelRunner stores as key in _model_update_group dict
    """
    import copy
    pp_meta = copy.copy(base_meta)
    pp_meta.nccl_master_port = port
    pp_meta.nccl_group_name = group_name
    logger.info(
        "[_build_pp_meta] pp_rank=%d port=%d group_name='%s' tp_size=%d",
        pp_rank, port, group_name, tp_size,
    )
    return pp_meta


@trace_perf("megatron_engine.pp_update_weights_from_distributed", category="comm")
def pp_update_weights_from_distributed(self: "MegatronEngine", meta: WeightUpdateMeta) -> None:
    """Update weights using per-PP-rank NCCL groups.

    For each parameter:
    1. Determine which SGLang PP rank owns this parameter
    2. Broadcast only to that PP rank's NCCL group
    3. SGLang workers in other PP ranks never participate
    """
    assert meta.gen_allocation is not None
    gen_parallel = meta.gen_allocation.parallel
    sglang_pp_size = gen_parallel.pp_size
    sglang_tp_size = gen_parallel.tp_size

    # Restore address from stored value
    meta.nccl_master_address = self.weight_update_master_addr

    if dist.get_rank() == 0:
        self.rollout_engine.pause_generation()

    dist.barrier(group=self.cpu_group)

    if self.is_pipeline_parallel_head():
        # Compute layer distribution: how SGLang splits layers across PP ranks
        total_layers = self.hf_config.num_hidden_layers
        layers_per_pp = _compute_layer_ranges(total_layers, sglang_pp_size)
        logger.info(
            "[pp_update] Layer distribution across %d PP ranks: %s (total_layers=%d)",
            sglang_pp_size, layers_per_pp, total_layers,
        )

        # Per-PP-rank buckets
        num_moe_experts = self.tf_config.num_moe_experts
        weight_chunked_mem_size = meta.weight_chunked_mem_mb * 1024 * 1024

        pp_buckets: dict[int, tuple[list, int]] = {
            pp_rank: ([], 0) for pp_rank in range(sglang_pp_size)
        }

        model_name = self.hf_config.model_type
        if self.config.use_lora:
            model_name = f"{model_name}_lora"

        for name, param in get_named_parameters(self.model, num_moe_experts):
            if ".experts." in name:
                continue
            if self.config.use_lora and (
                ".adapter." not in name or not getattr(param, "requires_grad", False)
            ):
                continue

            param, param_size = self._collect_param(name, param)

            # Determine which PP rank owns this parameter
            target_pp_rank = _get_param_pp_rank(name, layers_per_pp, sglang_pp_size)

            converted = convert_to_hf(
                self.tf_config,
                model_name,
                name,
                param,
                quantization_config=self.quantization_config,
                fp8_direct_convert=self.fp8_direct_convert,
            )

            bucket_tensors, bucket_size = pp_buckets[target_pp_rank]
            bucket_tensors.extend(converted)
            bucket_size += param_size

            # Flush if bucket is full
            if bucket_size > weight_chunked_mem_size:
                logger.info(
                    "[pp_update] Flushing bucket for pp_rank=%d: %d tensors, %.2f MB",
                    target_pp_rank, len(bucket_tensors), bucket_size / 1024 / 1024,
                )
                _pp_broadcast_bucket(self, meta, target_pp_rank, bucket_tensors)
                pp_buckets[target_pp_rank] = ([], 0)
            else:
                pp_buckets[target_pp_rank] = (bucket_tensors, bucket_size)

        # Flush remaining buckets
        for pp_rank in range(sglang_pp_size):
            bucket_tensors, bucket_size = pp_buckets[pp_rank]
            if bucket_tensors:
                logger.info(
                    "[pp_update] Flushing final bucket for pp_rank=%d: %d tensors, %.2f MB",
                    pp_rank, len(bucket_tensors), bucket_size / 1024 / 1024,
                )
                _pp_broadcast_bucket(self, meta, pp_rank, bucket_tensors)

        logger.info("[pp_update] All per-PP-rank weight broadcasts complete.")

    dist.barrier(group=self.cpu_group)

    # MoE expert weights
    if self.is_pipeline_parallel_head():
        _pp_update_expert_weights(self, meta, sglang_pp_size)

    dist.barrier(group=self.cpu_group)

    if dist.get_rank() == 0:
        self.rollout_engine.continue_generation()

    current_platform.synchronize()
    dist.barrier(group=self.cpu_group)


def _pp_broadcast_bucket(
    engine: "MegatronEngine",
    meta: WeightUpdateMeta,
    pp_rank: int,
    converted_named_tensors: list[tuple[str, nn.Parameter | torch.Tensor]],
) -> None:
    """Broadcast a bucket of parameters to a specific PP rank's NCCL group.

    Uses the EXISTING engine.rollout_engine (RolloutCallback) which routes
    through /callback/update_weights_xccl -> controller -> workers -> SGLang.
    The SGLang scheduler filter ensures only matching PP workers participate.
    """
    if not converted_named_tensors:
        return

    engine.engine_lock.acquire()

    group_name = engine._pp_weight_update_group_names[pp_rank]
    group = engine._pp_weight_update_groups[pp_rank]

    param_specs = [
        ParamSpec(
            name=name,
            shape=tuple(tensor.shape),
            dtype=str(tensor.dtype).split("torch.")[1],
        )
        for name, tensor in converted_named_tensors
    ]

    # Build per-PP-rank meta with correct group_name and port
    pp_meta = WeightUpdateMeta(
        type="xccl",
        gen_allocation=meta.gen_allocation,
        nccl_master_address=meta.nccl_master_address,
        nccl_master_port=engine._pp_weight_update_master_ports[pp_rank],
        nccl_group_name=group_name,
        weight_chunked_mem_mb=meta.weight_chunked_mem_mb,
        use_lora=meta.use_lora,
        lora_name=meta.lora_name,
        peft_config=meta.peft_config,
    )

    if engine.config.use_lora:
        from areal.engine.megatron_utils.megatron_lora import get_vllm_lora_target_modules
        pp_meta.peft_config = {
            "r": engine.config.lora_rank,
            "lora_alpha": engine.config.lora_alpha,
            "target_modules": get_vllm_lora_target_modules(
                list(engine.config.target_modules or [])
            ),
            "bias": "none",
        }

    logger.info(
        "[pp_broadcast] pp_rank=%d group='%s' n_params=%d",
        pp_rank, group_name, len(param_specs),
    )

    # Non-blocking: tell rollout side to start receiving
    # This uses the EXISTING RolloutCallback -> /callback/update_weights_xccl
    fut = engine.rollout_engine.update_weights_from_distributed(pp_meta, param_specs)

    # Broadcast from Megatron (rank 0) to SGLang TP workers in this PP group
    handles = []
    for _, param_tensor in converted_named_tensors:
        handles.append(
            dist.broadcast(param_tensor.data, 0, group=group, async_op=True)
        )
    for handle in handles:
        handle.wait()

    # Wait for rollout side to finish receiving
    fut.result()

    converted_named_tensors.clear()
    engine.engine_lock.release()

    logger.info(
        "[pp_broadcast] pp_rank=%d group='%s' broadcast complete.", pp_rank, group_name,
    )


def _compute_layer_ranges(total_layers: int, pp_size: int) -> list[tuple[int, int]]:
    """Compute (start_layer, end_layer) for each PP rank.

    Uses the same even-split logic as SGLang's default PP layer distribution.
    end_layer is exclusive: [start, end).
    """
    layers_per_rank = total_layers // pp_size
    remainder = total_layers % pp_size
    ranges = []
    start = 0
    for rank in range(pp_size):
        count = layers_per_rank + (1 if rank < remainder else 0)
        ranges.append((start, start + count))
        start += count
    logger.info(
        "[_compute_layer_ranges] total_layers=%d, pp_size=%d, ranges=%s",
        total_layers, pp_size, ranges,
    )
    return ranges


def _get_param_pp_rank(
    param_name: str,
    layer_ranges: list[tuple[int, int]],
    pp_size: int,
) -> int:
    """Determine which SGLang PP rank owns a given parameter.

    Logic:
    - 'model.layers.{N}.*' -> PP rank whose range covers layer N
    - 'model.embed_tokens.*' -> PP rank 0 (first stage)
    - 'model.norm.*' / 'lm_head.*' -> last PP rank (last stage)
    - Unknown patterns -> PP rank 0 (safe default)
    """
    if "model.layers." in param_name:
        parts = param_name.split(".")
        try:
            layer_idx_pos = parts.index("layers") + 1
            layer_idx = int(parts[layer_idx_pos])
        except (ValueError, IndexError):
            logger.warning(
                "[_get_param_pp_rank] Cannot parse layer index from '%s', defaulting to pp_rank=0",
                param_name,
            )
            return 0

        for pp_rank, (start, end) in enumerate(layer_ranges):
            if start <= layer_idx < end:
                return pp_rank

        logger.warning(
            "[_get_param_pp_rank] Layer %d out of range for '%s', defaulting to last pp_rank=%d",
            layer_idx, param_name, pp_size - 1,
        )
        return pp_size - 1

    if "embed_tokens" in param_name:
        return 0

    if "model.norm" in param_name or "lm_head" in param_name:
        return pp_size - 1

    logger.warning(
        "[_get_param_pp_rank] Unknown param pattern '%s', defaulting to pp_rank=0",
        param_name,
    )
    return 0


def _pp_update_expert_weights(
    engine: "MegatronEngine",
    meta: WeightUpdateMeta,
    sglang_pp_size: int,
) -> None:
    """Handle MoE expert weight updates with per-PP-rank groups."""
    num_moe_experts = engine.tf_config.num_moe_experts
    if num_moe_experts is None:
        return

    total_layers = engine.hf_config.num_hidden_layers
    layer_ranges = _compute_layer_ranges(total_layers, sglang_pp_size)
    weight_chunked_mem_size = meta.weight_chunked_mem_mb * 1024 * 1024
    model_name = engine.hf_config.model_type

    pp_buckets: dict[int, tuple[list, int]] = {
        pp_rank: ([], 0) for pp_rank in range(sglang_pp_size)
    }

    for name, param in get_named_parameters(engine.model, num_moe_experts):
        if ".experts." not in name or engine.config.use_lora:
            continue

        param, param_size = engine._collect_param(name, param)
        target_pp_rank = _get_param_pp_rank(name, layer_ranges, sglang_pp_size)

        converted = convert_to_hf(
            engine.tf_config, model_name, name, param,
            quantization_config=engine.quantization_config,
            fp8_direct_convert=engine.fp8_direct_convert,
        )

        bucket_tensors, bucket_size = pp_buckets[target_pp_rank]
        bucket_tensors.extend(converted)
        bucket_size += param_size

        if bucket_size > weight_chunked_mem_size:
            _pp_broadcast_bucket(engine, meta, target_pp_rank, bucket_tensors)
            pp_buckets[target_pp_rank] = ([], 0)
        else:
            pp_buckets[target_pp_rank] = (bucket_tensors, bucket_size)

    for pp_rank in range(sglang_pp_size):
        bucket_tensors, _ = pp_buckets[pp_rank]
        if bucket_tensors:
            _pp_broadcast_bucket(engine, meta, pp_rank, bucket_tensors)


# ---------------------------------------------------------------------------
#  Installation: monkey-patch MegatronEngine
# ---------------------------------------------------------------------------

def install_pp_weight_update(engine_class):
    """Install per-PP-rank weight update methods on MegatronEngine class."""
    engine_class._orig_init_weight_update_from_distributed = (
        engine_class._init_weight_update_from_distributed
    )
    engine_class._orig_update_weights_from_distributed = (
        engine_class._update_weights_from_distributed
    )

    engine_class._init_weight_update_from_distributed = pp_init_weight_update_from_distributed
    engine_class._update_weights_from_distributed = pp_update_weights_from_distributed

    logger.info(
        "[install_pp_weight_update] Installed per-PP-rank weight update on %s",
        engine_class.__name__,
    )
