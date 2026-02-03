from __future__ import annotations

import os
from concurrent.futures import Future
from datetime import datetime
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DTensor

from areal.api.io_struct import ParamSpec, WeightUpdateMeta
from areal.experimental.engine.archon_checkpoint import save_model_to_hf
from areal.infra.platforms import current_platform
from areal.utils import name_resolve, names
from areal.utils.constants import DIST_GROUP_DEFAULT_TIMEOUT
from areal.utils.distributed import init_custom_process_group
from areal.utils.lock import DistributedLock
from areal.utils.network import find_free_ports, gethostip
from areal.utils.perf_tracer import trace_perf

if TYPE_CHECKING:
    from areal.api.engine_api import InferenceEngine
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
        assert meta.alloc_mode is not None

        engine.engine_lock.acquire()

        fut = engine.rollout_engine.init_weights_update_group(meta)

        engine.logger.info(
            f"Initializing weight update group: type={meta.type}, "
            f"init_method=tcp://{meta.nccl_master_address}:{meta.nccl_master_port}, "
            f"group={meta.nccl_group_name}"
        )
        state.group = init_custom_process_group(
            backend=current_platform.communication_backend,
            world_size=meta.alloc_mode.gen.world_size + 1,
            init_method=f"tcp://{meta.nccl_master_address}:{meta.nccl_master_port}",
            rank=0,
            group_name=meta.nccl_group_name,
            timeout=DIST_GROUP_DEFAULT_TIMEOUT,
        )

        fut.result()

        engine.engine_lock.release()

    state.group_initialized = True


def _get_full_tensor(param: nn.Parameter) -> torch.Tensor:
    """Get full tensor from a parameter, handling DTensor and CPU offload."""
    tensor = param.data
    if isinstance(tensor, DTensor):
        if tensor.device.type != "cpu":
            return tensor.full_tensor()

        temp_dtensor = DTensor.from_local(
            tensor.to_local(),
            device_mesh=tensor.device_mesh,
            placements=tensor.placements,
        )
        return temp_dtensor.full_tensor()
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
    """Update weights by broadcasting from training engine to inference engine."""
    assert engine.rollout_engine is not None

    meta.nccl_master_address = state.master_addr
    meta.nccl_master_port = state.master_port
    meta.nccl_group_name = state.group_name

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

    engine_lock.acquire()

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
        handles.append(dist.broadcast(tensor, src=0, group=state.group, async_op=True))
    for handle in handles:
        handle.wait()

    fut.result()

    named_tensors.clear()

    engine_lock.release()


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
