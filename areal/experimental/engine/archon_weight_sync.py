# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import time
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
        self.pp_rank = pp_rank
        self.group_initialized: bool = False
        self.group_version: int = 0
        self.group_name: str = self._make_group_name()
        self.master_addr: str = ""
        self.master_port: int = 0
        self.group: dist.ProcessGroup | None = None
        self.active_server_addresses: list[str] = []

    def _make_group_name(self) -> str:
        return f"update_weight_group_{self.pp_rank}_v{self.group_version}"


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
    meta.group_version = state.group_version
    if not meta.active_server_addresses:
        meta.active_server_addresses = list(state.active_server_addresses)
    if not meta.target_server_addresses:
        meta.target_server_addresses = list(state.active_server_addresses)

    # Processes launched with torchrun set TORCHELASTIC_USE_AGENT_STORE=True,
    # which blocks creating another TCP store for weight update.
    os.environ["TORCHELASTIC_USE_AGENT_STORE"] = str(False)

    if engine.is_pipeline_parallel_head():
        assert meta.gen_allocation is not None

        with engine.engine_lock:
            target_addresses = list(meta.target_server_addresses)
            fut = engine.rollout_engine.init_weights_update_group(meta)

            gen_world_size = len(target_addresses)
            init_method = f"tcp://{format_host_for_url(meta.nccl_master_address)}:{meta.nccl_master_port}"
            engine.logger.info(
                f"Initializing weight update group: type={meta.type}, "
                f"init_method={init_method}, "
                f"group={meta.nccl_group_name}, "
                f"servers={target_addresses}"
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


def _get_weight_update_fileroot(engine: ArchonEngine) -> str | None:
    return getattr(engine.rollout_engine.config, "fileroot", None)


def _should_delay_group_rebuild(engine: ArchonEngine) -> bool:
    if engine.rollout_engine is None or not hasattr(
        engine.rollout_engine, "get_last_topology_change_time"
    ):
        return False
    cooldown = getattr(
        engine.rollout_engine.config, "topology_change_cooldown_seconds", 0.0
    )
    if cooldown <= 0:
        return False
    last_change = engine.rollout_engine.get_last_topology_change_time()
    return (time.time() - last_change) < cooldown


def _sync_new_servers_from_disk(
    state: WeightSyncState,
    meta: WeightUpdateMeta,
    engine: ArchonEngine,
    new_addresses: list[str],
) -> None:
    if not new_addresses or dist.get_rank() != 0:
        return
    if not engine.config.experiment_name or not engine.config.trial_name:
        engine.logger.warning(
            "Skip disk fallback for new inference servers because experiment/trial metadata is missing."
        )
        return
    fileroot = _get_weight_update_fileroot(engine)
    if not fileroot:
        engine.logger.warning(
            "Skip disk fallback for new inference servers because fileroot is not configured."
        )
        return

    disk_meta = WeightUpdateMeta.from_disk(
        experiment_name=engine.config.experiment_name,
        trial_name=engine.config.trial_name,
        file_root=fileroot,
        use_lora=meta.use_lora,
        lora_name=meta.lora_name,
        lora_int_id=meta.lora_int_id,
        base_model_name=meta.base_model_name,
        clear_checkpoint_after_load=True,
    ).with_version(engine.get_version())
    disk_meta.target_server_addresses = list(new_addresses)
    engine.logger.info(
        "Sync %d newly registered inference servers via disk fallback before rebuilding %s.",
        len(new_addresses),
        state.group_name,
    )
    update_weights_from_disk(disk_meta, engine)


def maybe_rebuild_weight_update_group(
    state: WeightSyncState,
    meta: WeightUpdateMeta,
    engine: ArchonEngine,
) -> None:
    latest_addresses = list(state.active_server_addresses)
    if engine.rollout_engine is not None and hasattr(
        engine.rollout_engine, "get_active_server_addresses"
    ):
        latest_addresses = list(engine.rollout_engine.get_active_server_addresses())

    if not latest_addresses:
        return

    topology_changed = latest_addresses != state.active_server_addresses
    if dist.get_rank() == 0:
        remote_flag = (
            engine.rollout_engine.consume_group_rebuild_request()
            if engine.rollout_engine is not None
            and hasattr(engine.rollout_engine, "consume_group_rebuild_request")
            else False
        )
        topology_changed = topology_changed or remote_flag
    topology_changed_list = [topology_changed]
    dist.broadcast_object_list(topology_changed_list, src=0, group=engine.cpu_group)
    topology_changed = bool(topology_changed_list[0])

    if not topology_changed:
        state.active_server_addresses = list(latest_addresses)
        meta.active_server_addresses = list(latest_addresses)
        meta.target_server_addresses = list(latest_addresses)
        return

    new_addresses = [
        addr for addr in latest_addresses if addr not in state.active_server_addresses
    ]
    removed_addresses = [
        addr for addr in state.active_server_addresses if addr not in latest_addresses
    ]
    if new_addresses:
        _sync_new_servers_from_disk(state, meta, engine, new_addresses)

    if new_addresses and not removed_addresses and _should_delay_group_rebuild(engine):
        state.active_server_addresses = list(latest_addresses)
        meta.active_server_addresses = list(latest_addresses)
        meta.target_server_addresses = list(latest_addresses)
        engine.logger.info(
            "Delayed XCCL group rebuild for %d newly added inference servers due to topology cooldown.",
            len(new_addresses),
        )
        return

    dist.barrier(group=engine.cpu_group)
    if engine.is_pipeline_parallel_head() and state.group_initialized and state.group:
        with engine.engine_lock:
            dist.destroy_process_group(state.group)
        state.group = None
        state.group_initialized = False
    dist.barrier(group=engine.cpu_group)

    state.group_version += 1
    state.group_name = state._make_group_name()
    state.active_server_addresses = list(latest_addresses)
    meta.group_version = state.group_version
    meta.active_server_addresses = list(latest_addresses)
    meta.target_server_addresses = list(latest_addresses)
    init_weight_update_group(state=state, meta=meta, engine=engine)


@trace_perf("archon_engine.update_weights_from_distributed", category="comm")
def update_weights_from_distributed(
    state: WeightSyncState,
    meta: WeightUpdateMeta,
    engine: ArchonEngine,
) -> None:
    """Update weights by broadcasting from training engine to inference engine."""
    assert engine.rollout_engine is not None

    maybe_rebuild_weight_update_group(state=state, meta=meta, engine=engine)

    meta.nccl_master_address = state.master_addr
    meta.nccl_master_port = state.master_port
    meta.nccl_group_name = state.group_name
    meta.group_version = state.group_version
    meta.active_server_addresses = list(state.active_server_addresses)
    meta.target_server_addresses = list(state.active_server_addresses)

    if dist.get_rank() == 0:
        engine.rollout_engine.pause_generation()

    dist.barrier(group=engine.cpu_group)

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
