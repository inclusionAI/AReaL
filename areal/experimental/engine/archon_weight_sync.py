from __future__ import annotations

import copy
import logging
import os
from concurrent.futures import Future
from datetime import datetime
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DTensor

from areal.api import ParamSpec, WeightUpdateMeta
from areal.engine.core.distributed import init_custom_process_group
from areal.experimental.engine.archon_checkpoint import save_model_to_hf
from areal.infra.platforms import current_platform
from areal.utils import name_resolve, names
from areal.utils.constants import DIST_GROUP_DEFAULT_TIMEOUT
from areal.utils.lock import DistributedLock
from areal.utils.network import find_free_ports, format_host_for_url, gethostip
from areal.utils.perf_tracer import trace_perf

if TYPE_CHECKING:
    from areal.api import InferenceEngine
    from areal.experimental.engine.archon_engine import ArchonEngine

logger = logging.getLogger(__name__)


class WeightSyncState:
    """State container for weight synchronization.

    Attributes:
        group_initialized: Whether the weight update group has been initialized.
        group_name: Name of the NCCL group for weight updates (single-group mode).
        master_addr: Master address for TCP store initialization (single-group mode).
        master_port: Master port for TCP store initialization (single-group mode).
        group: The distributed process group for weight updates (single-group mode).
        group_names: List of group names for per-PP-rank groups (multi-group mode).
        groups: List of process groups for per-PP-rank groups (multi-group mode).
        master_addrs: List of master addresses for per-PP-rank groups (multi-group mode).
        master_ports: List of master ports for per-PP-rank groups (multi-group mode).
    """

    def __init__(self, pp_rank: int):
        self.group_initialized: bool = False
        self.group_name: str = f"update_weight_group_{pp_rank}"
        self.master_addr: str = ""
        self.master_port: int = 0
        self.group: dist.ProcessGroup | None = None
        # Multi-group state for sglang PP > 1
        self.group_names: list[str] = []
        self.groups: list[dist.ProcessGroup] = []
        self.master_addrs: list[str] = []
        self.master_ports: list[int] = []


def init_weight_update_group(
    state: WeightSyncState,
    meta: WeightUpdateMeta,
    engine: ArchonEngine,
) -> None:
    """Initialize the weight update process group for XCCL synchronization.

    When the inference engine (sglang) uses pipeline parallelism (gen_pp_size > 1),
    creates per-PP-rank NCCL groups so that each sglang PP stage receives only
    the parameters it owns. When gen_pp_size == 1, falls back to the original
    single-group path.
    """
    assert meta.type == "xccl"

    # Processes launched with torchrun set TORCHELASTIC_USE_AGENT_STORE=True,
    # which blocks creating another TCP store for weight update.
    os.environ["TORCHELASTIC_USE_AGENT_STORE"] = str(False)

    gen_pp_size = meta.gen_allocation.parallel.pp_size if meta.gen_allocation else 1
    logger.info(
        f"[ArchonWeightSync] init_weight_update_group called: "
        f"gen_pp_size={gen_pp_size}, is_pp_head={engine.is_pipeline_parallel_head()}, "
        f"group_name={state.group_name}"
    )

    if gen_pp_size > 1:
        _init_per_pp_weight_update_groups(state, meta, engine, gen_pp_size)
    else:
        _init_single_weight_update_group(state, meta, engine)

    state.group_initialized = True
    logger.info(
        f"[ArchonWeightSync] Weight update group(s) initialized successfully. "
        f"single_group={state.group is not None}, "
        f"multi_groups={len(state.groups)}"
    )


def _init_single_weight_update_group(
    state: WeightSyncState,
    meta: WeightUpdateMeta,
    engine: ArchonEngine,
) -> None:
    """Original single-group initialization path (gen_pp_size == 1)."""
    state.master_addr = gethostip()
    state.master_port = find_free_ports(1)[0]

    meta.nccl_master_address = state.master_addr
    meta.nccl_master_port = state.master_port
    meta.nccl_group_name = state.group_name

    if engine.is_pipeline_parallel_head():
        assert meta.gen_allocation is not None

        with engine.engine_lock:
            fut = engine.rollout_engine.init_weights_update_group(meta)

            gen_world_size = meta.gen_allocation.parallel.world_size
            init_method = f"tcp://{format_host_for_url(meta.nccl_master_address)}:{meta.nccl_master_port}"
            engine.logger.info(
                f"[ArchonWeightSync] Initializing single weight update group: "
                f"type={meta.type}, init_method={init_method}, "
                f"group={meta.nccl_group_name}, world_size={gen_world_size + 1}"
            )
            state.group = init_custom_process_group(
                backend=current_platform.communication_backend,
                world_size=gen_world_size + 1,
                init_method=init_method,
                rank=0,
                group_name=meta.nccl_group_name,
                timeout=DIST_GROUP_DEFAULT_TIMEOUT,
            )

            fut.result()
            engine.logger.info(
                f"[ArchonWeightSync] Single weight update group initialized: "
                f"group={meta.nccl_group_name}"
            )


def _init_per_pp_weight_update_groups(
    state: WeightSyncState,
    meta: WeightUpdateMeta,
    engine: ArchonEngine,
    gen_pp_size: int,
) -> None:
    """Create per-PP-rank NCCL groups for sglang PP > 1.

    Each sglang PP rank gets its own NCCL group. The archon training engine
    (which is the single source of full model weights) joins ALL groups as rank 0,
    broadcasting the appropriate weight slices to each PP stage.

    This matches the pattern used by FSDP and Megatron engines.
    """
    if not engine.is_pipeline_parallel_head():
        logger.info(
            "[ArchonWeightSync] Non-PP-head rank, skipping per-PP group creation."
        )
        return

    assert meta.gen_allocation is not None
    gen_world_size = meta.gen_allocation.parallel.world_size
    # per_pp_world_size = total_gen_workers / pp_stages
    # This equals n_servers * tp_size (workers at each PP stage)
    per_pp_world_size = gen_world_size // gen_pp_size

    engine.logger.info(
        f"[ArchonWeightSync] Creating per-PP-rank groups: gen_pp_size={gen_pp_size}, "
        f"gen_world_size={gen_world_size}, "
        f"per_pp_world_size={per_pp_world_size}, "
        f"tp_size={meta.gen_allocation.parallel.tp_size}"
    )

    host_addr = gethostip()
    free_ports = find_free_ports(gen_pp_size)

    for pp_rank in range(gen_pp_size):
        group_name = f"update_weight_group_{pp_rank}"
        master_addr = host_addr
        master_port = free_ports[pp_rank]
        world_size = per_pp_world_size + 1  # +1 for the training engine rank

        state.group_names.append(group_name)
        state.master_addrs.append(master_addr)
        state.master_ports.append(master_port)

        pp_meta = copy.copy(meta)
        pp_meta.nccl_master_address = master_addr
        pp_meta.nccl_master_port = master_port
        pp_meta.nccl_group_name = group_name

        init_method = f"tcp://{format_host_for_url(master_addr)}:{master_port}"
        engine.logger.info(
            f"[ArchonWeightSync] Initializing per-PP group: pp_rank={pp_rank}, "
            f"group={group_name}, init_method={init_method}, "
            f"world_size={world_size}"
        )

        with engine.engine_lock:
            fut = engine.rollout_engine.init_weights_update_group(pp_meta)

            group = init_custom_process_group(
                backend=current_platform.communication_backend,
                world_size=world_size,
                init_method=init_method,
                rank=0,
                group_name=group_name,
                timeout=DIST_GROUP_DEFAULT_TIMEOUT,
            )
            state.groups.append(group)

            fut.result()

        engine.logger.info(
            f"[ArchonWeightSync] Per-PP group initialized: pp_rank={pp_rank}, "
            f"group={group_name}"
        )

    # Set single-group aliases to the first group for backward compatibility
    state.group_name = state.group_names[0]
    state.master_addr = state.master_addrs[0]
    state.master_port = state.master_ports[0]
    state.group = state.groups[0]


def _get_full_tensor(param: nn.Parameter) -> torch.Tensor:
    """Get full tensor from a parameter, handling DTensor and CPU offload."""
    tensor = param.data
    if isinstance(tensor, DTensor):
        if tensor.device.type != "cpu":
            return tensor.full_tensor()

        return DTensor.from_local(
            tensor.to_local(),
            device_mesh=tensor.device_mesh,
            placements=tensor.placements,
        ).full_tensor()
    else:
        if tensor.device.type == "cpu":
            tensor = tensor.to(current_platform.device_type)
        return tensor


@trace_perf("archon_engine.update_weights_from_distributed", category="comm")
def update_weights_from_distributed(
    state: WeightSyncState,
    meta: WeightUpdateMeta,
    engine: ArchonEngine,
) -> None:
    """Update weights by broadcasting from training engine to inference engine.

    When multi-group mode is active (sglang PP > 1), broadcasts weights to
    each per-PP-rank group sequentially. Otherwise uses the original single-group path.
    """
    assert engine.rollout_engine is not None

    if dist.get_rank() == 0:
        engine.rollout_engine.pause_generation()

    dist.barrier(group=engine.cpu_group)

    if len(state.groups) > 1:
        logger.info(
            f"[ArchonWeightSync] update_weights: multi-group mode, "
            f"{len(state.groups)} groups"
        )
        _update_weights_multi_group(state, meta, engine)
    else:
        logger.info("[ArchonWeightSync] update_weights: single-group mode")
        meta.nccl_master_address = state.master_addr
        meta.nccl_master_port = state.master_port
        meta.nccl_group_name = state.group_name
        _update_weights_single_group(state, meta, engine)

    dist.barrier(group=engine.cpu_group)

    if dist.get_rank() == 0:
        engine.rollout_engine.continue_generation()

    current_platform.synchronize()
    dist.barrier(group=engine.cpu_group)


def _update_weights_single_group(
    state: WeightSyncState,
    meta: WeightUpdateMeta,
    engine: ArchonEngine,
) -> None:
    """Original single-group weight update path."""
    weight_chunked_mem_size = meta.weight_chunked_mem_mb * 1024 * 1024

    buffer_size = 0
    named_tensors: list[tuple[str, torch.Tensor]] = []

    for name, param in engine._get_model_name_parameters():
        tensor = _get_full_tensor(param)

        if not engine.is_pipeline_parallel_head():
            continue

        if engine.state_dict_adapter is not None:
            hf_pairs = engine.state_dict_adapter.convert_single_to_hf(name, tensor)
        else:
            hf_pairs = [(name, tensor)]

        for hf_name, hf_tensor in hf_pairs:
            tensor_size = hf_tensor.numel() * hf_tensor.element_size()

            if tensor_size + buffer_size > weight_chunked_mem_size:
                _update_bucket_weights(
                    state,
                    meta,
                    engine.rollout_engine,
                    engine.engine_lock,
                    named_tensors,
                )
                buffer_size = 0
                named_tensors = []

            named_tensors.append((hf_name, hf_tensor))
            buffer_size += tensor_size

    if named_tensors:
        _update_bucket_weights(
            state, meta, engine.rollout_engine, engine.engine_lock, named_tensors
        )


def _update_weights_multi_group(
    state: WeightSyncState,
    meta: WeightUpdateMeta,
    engine: ArchonEngine,
) -> None:
    """Multi-group weight update path for sglang PP > 1.

    Broadcasts all weights to each per-PP-rank group sequentially.
    Each sglang PP rank receives all parameters; the sglang side is responsible
    for filtering which parameters belong to its PP stage.
    """
    weight_chunked_mem_size = meta.weight_chunked_mem_mb * 1024 * 1024

    # Collect all named tensors first
    all_named_tensors: list[tuple[str, torch.Tensor]] = []

    for name, param in engine._get_model_name_parameters():
        tensor = _get_full_tensor(param)

        if not engine.is_pipeline_parallel_head():
            continue

        if engine.state_dict_adapter is not None:
            hf_pairs = engine.state_dict_adapter.convert_single_to_hf(name, tensor)
        else:
            hf_pairs = [(name, tensor)]

        for hf_name, hf_tensor in hf_pairs:
            all_named_tensors.append((hf_name, hf_tensor))

    logger.info(
        f"[ArchonWeightSync] Multi-group update: {len(all_named_tensors)} tensors "
        f"to broadcast to {len(state.groups)} groups"
    )

    # Broadcast to each PP group sequentially
    for group_idx in range(len(state.groups)):
        group_name = state.group_names[group_idx]
        group = state.groups[group_idx]
        master_addr = state.master_addrs[group_idx]
        master_port = state.master_ports[group_idx]

        pp_meta = copy.copy(meta)
        pp_meta.nccl_master_address = master_addr
        pp_meta.nccl_master_port = master_port
        pp_meta.nccl_group_name = group_name

        logger.info(
            f"[ArchonWeightSync] Broadcasting to group {group_idx}: "
            f"group={group_name}"
        )

        buffer_size = 0
        named_tensors: list[tuple[str, torch.Tensor]] = []

        for hf_name, hf_tensor in all_named_tensors:
            tensor_size = hf_tensor.numel() * hf_tensor.element_size()

            if tensor_size + buffer_size > weight_chunked_mem_size:
                _update_bucket_weights_multi_group(
                    group,
                    pp_meta,
                    engine.rollout_engine,
                    engine.engine_lock,
                    named_tensors,
                )
                buffer_size = 0
                named_tensors = []

            named_tensors.append((hf_name, hf_tensor))
            buffer_size += tensor_size

        if named_tensors:
            _update_bucket_weights_multi_group(
                group,
                pp_meta,
                engine.rollout_engine,
                engine.engine_lock,
                named_tensors,
            )

        logger.info(
            f"[ArchonWeightSync] Finished broadcasting to group {group_idx}: "
            f"group={group_name}"
        )


def _update_bucket_weights(
    state: WeightSyncState,
    meta: WeightUpdateMeta,
    rollout_engine: InferenceEngine,
    engine_lock: DistributedLock,
    named_tensors: list[tuple[str, torch.Tensor]],
) -> None:
    """Broadcast a bucket of weights to the inference engine (single-group mode)."""
    if not named_tensors:
        return

    with engine_lock:
        param_specs = [
            ParamSpec(
                name=name,
                shape=tuple(tensor.shape),
                dtype=str(tensor.dtype).split("torch.")[1],
            )
            for name, tensor in named_tensors
        ]

        fut = rollout_engine.update_weights_from_distributed(meta, param_specs)

        handles = []
        assert state.group is not None
        for _, tensor in named_tensors:
            handles.append(
                dist.broadcast(tensor, src=0, group=state.group, async_op=True)
            )
        for handle in handles:
            handle.wait()

        fut.result()

        named_tensors.clear()


def _update_bucket_weights_multi_group(
    group: dist.ProcessGroup,
    meta: WeightUpdateMeta,
    rollout_engine: InferenceEngine,
    engine_lock: DistributedLock,
    named_tensors: list[tuple[str, torch.Tensor]],
) -> None:
    """Broadcast a bucket of weights to one per-PP-rank group (multi-group mode)."""
    if not named_tensors:
        return

    with engine_lock:
        param_specs = [
            ParamSpec(
                name=name,
                shape=tuple(tensor.shape),
                dtype=str(tensor.dtype).split("torch.")[1],
            )
            for name, tensor in named_tensors
        ]

        fut = rollout_engine.update_weights_from_distributed(meta, param_specs)

        handles = []
        for _, tensor in named_tensors:
            handles.append(
                dist.broadcast(tensor, src=0, group=group, async_op=True)
            )
        for handle in handles:
            handle.wait()

        fut.result()

        named_tensors.clear()


@trace_perf("archon_engine.update_weights_from_disk", category="io")
def update_weights_from_disk(
    meta: WeightUpdateMeta,
    engine: ArchonEngine,
) -> None:
    """Update weights by saving to disk and loading in inference engine."""
    fut: Future | None = None

    if dist.get_rank() == 0:
        fut = engine.rollout_engine.update_weights_from_disk(meta)

    assert meta.path is not None
    save_model_to_hf(engine, meta.path, engine.tokenizer, None)

    if dist.get_rank() == 0:
        update_name = names.update_weights_from_disk(
            engine.config.experiment_name,
            engine.config.trial_name,
            engine.get_version(),
        )
        name_resolve.add(
            update_name, str(datetime.now().timestamp()), keepalive_ttl=120
        )

        assert fut is not None
        fut.result()

    current_platform.synchronize()
    dist.barrier(group=engine.cpu_group)
