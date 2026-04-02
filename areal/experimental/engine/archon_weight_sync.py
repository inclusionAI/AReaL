from __future__ import annotations

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
from areal.engine.core.tensor_hash import hash_parameter_shards, hash_string_sequence
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


class WeightSyncState:
    """State container for weight synchronization.

    Attributes:
        group_initialized: Whether the weight update group has been initialized.
        group_name: Name of the NCCL group for weight updates.
        master_addr: Master address for TCP store initialization.
        master_port: Master port for TCP store initialization.
        group: The distributed process group for weight updates.
    """

    def __init__(self, pp_rank: int):
        self.group_initialized: bool = False
        self.group_name: str = f"update_weight_group_{pp_rank}"
        self.master_addr: str = ""
        self.master_port: int = 0
        self.group: dist.ProcessGroup | None = None
        # Hash-based sparse weight update state
        self.param_hashes: torch.Tensor | None = None
        self.weight_sync_groups: tuple[dist.ProcessGroup, ...] = ()


def init_weight_update_group(
    state: WeightSyncState,
    meta: WeightUpdateMeta,
    engine: ArchonEngine,
) -> None:
    """Initialize the weight update process group for XCCL synchronization."""
    assert meta.type == "xccl"

    state.master_addr = gethostip()
    state.master_port = find_free_ports(1)[0]

    meta.nccl_master_address = state.master_addr
    meta.nccl_master_port = state.master_port
    meta.nccl_group_name = state.group_name

    # Processes launched with torchrun set TORCHELASTIC_USE_AGENT_STORE=True,
    # which blocks creating another TCP store for weight update.
    os.environ["TORCHELASTIC_USE_AGENT_STORE"] = str(False)

    if engine.is_pipeline_parallel_head():
        assert meta.gen_allocation is not None

        with engine.engine_lock:
            fut = engine.rollout_engine.init_weights_update_group(meta)

            gen_world_size = meta.gen_allocation.parallel.world_size
            init_method = f"tcp://{format_host_for_url(meta.nccl_master_address)}:{meta.nccl_master_port}"
            engine.logger.info(
                f"Initializing weight update group: type={meta.type}, "
                f"init_method={init_method}, "
                f"group={meta.nccl_group_name}"
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

    state.group_initialized = True


def build_weight_sync_groups(engine: ArchonEngine) -> tuple[dist.ProcessGroup, ...]:
    """Build process groups for hash-based change detection all-reduce.

    Uses two-phase all-reduce on orthogonal mesh dimensions (same pattern
    as grad norm computation, adapted from Megatron-LM).

    For Archon with PP: each PP stage independently manages its own hashes.
    All ranks within the same PP stage share identical named_parameters()
    ordering (they hold the same model_parts with only DTensor shard differences).
    EP ranks also share identical param names (experts are DTensor Shard(0),
    not separate modules).
    """
    groups: list[dist.ProcessGroup] = []
    dp_shard_cp_group = engine.parallel_dims.get_group("dp_shard_cp")
    if dp_shard_cp_group is not None:
        groups.append(dp_shard_cp_group)
    tp_group = engine.parallel_dims.get_group("tp")
    if engine.parallel_dims.tp_enabled and tp_group is not None:
        groups.append(tp_group)
    return tuple(groups)


def init_param_hashes(
    state: WeightSyncState,
    engine: ArchonEngine,
) -> None:
    """Initialize parameter hashes after model loading.

    Called once during connect_engine() regardless of sparse_weight_sync.

    Always performs ordering verification (collective-safety invariant):
    all ranks in the weight sync groups must see identical parameter names
    in the same order, because update_weights_from_distributed() iterates
    param_list and calls DTensor.full_tensor() (a collective) in that order.

    Hash caching for incremental detection is only done when
    sparse_weight_sync is enabled.
    """
    param_list = list(engine._get_model_name_parameters())

    # One-time ordering verification across ranks (unconditional).
    param_names = tuple(name for name, _ in param_list)
    name_hash_tensor = hash_string_sequence(
        param_names, output_device=current_platform.current_device()
    )
    for group in state.weight_sync_groups:
        gathered = [
            torch.zeros_like(name_hash_tensor)
            for _ in range(dist.get_world_size(group=group))
        ]
        dist.all_gather(gathered, name_hash_tensor, group=group)
        if not all(torch.equal(g, name_hash_tensor) for g in gathered):
            raise RuntimeError(
                f"Parameter ordering mismatch across ranks in weight sync group. "
                f"This rank sees {len(param_names)} params starting with "
                f"{list(param_names[:3])}"
            )

    # Hash caching only needed for sparse weight sync.
    if not engine.config.sparse_weight_sync:
        engine.logger.info(
            f"sparse_weight_sync=False: skipping hash caching for "
            f"{len(param_list)} parameters."
        )
        return

    state.param_hashes = hash_parameter_shards(
        param_list, output_device=current_platform.current_device()
    )
    engine.logger.info(
        f"Initialized param hashes for {len(param_list)} parameters "
        f"(hash shape: {state.param_hashes.shape})."
    )


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
    """Update weights with hash-based skip for unchanged params."""
    assert engine.rollout_engine is not None

    meta.nccl_master_address = state.master_addr
    meta.nccl_master_port = state.master_port
    meta.nccl_group_name = state.group_name

    if dist.get_rank() == 0:
        engine.rollout_engine.pause_generation()

    dist.barrier(group=engine.cpu_group)

    # ── Pass 1: Hash-based change detection (no collectives) ──
    param_list = list(engine._get_model_name_parameters())
    n = len(param_list)

    if not engine.config.sparse_weight_sync:
        changed_list = [1] * n
        num_changed = n
        new_hashes = None
    else:
        new_hashes = hash_parameter_shards(
            param_list, output_device=current_platform.current_device()
        )

        if state.param_hashes is None:
            changed_list = [1] * n
            num_changed = n
            engine.logger.info(
                "Full weight sync forced (param hashes cleared after engine reconnect)."
            )
        else:
            if state.param_hashes.shape != new_hashes.shape:
                raise RuntimeError(
                    "Parameter hash shape changed after initialization. "
                    f"Expected {state.param_hashes.shape}, got {new_hashes.shape}."
                )
            changed = (new_hashes != state.param_hashes).to(torch.int32)

            # Two-phase all-reduce on orthogonal dims
            for group in state.weight_sync_groups:
                dist.all_reduce(changed, op=dist.ReduceOp.MAX, group=group)

            changed_list = changed.tolist()
            num_changed = int(changed.sum().item())

    engine.logger.info(
        f"Weight update: {num_changed}/{n} params changed "
        f"({100 * num_changed / max(n, 1):.1f}%)"
    )

    # ── Pass 2: Selective weight update (only changed params) ──

    weight_chunked_mem_size = meta.weight_chunked_mem_mb * 1024 * 1024

    buffer_size = 0
    named_tensors: list[tuple[str, torch.Tensor]] = []

    for i, (name, param) in enumerate(param_list):
        if changed_list[i] == 0:
            continue  # All ranks skip together — no deadlock

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

    # Commit hashes only after all buckets complete successfully.
    # If any broadcast/future failed above, the exception propagates
    # past this point, preserving old hashes so the next retry
    # re-sends those parameters.
    if new_hashes is not None:
        state.param_hashes = new_hashes

    dist.barrier(group=engine.cpu_group)

    if dist.get_rank() == 0:
        engine.rollout_engine.continue_generation()

    current_platform.synchronize()
    dist.barrier(group=engine.cpu_group)


def _update_bucket_weights(
    state: WeightSyncState,
    meta: WeightUpdateMeta,
    rollout_engine: InferenceEngine,
    engine_lock: DistributedLock,
    named_tensors: list[tuple[str, torch.Tensor]],
) -> None:
    """Broadcast a bucket of weights to the inference engine."""
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
