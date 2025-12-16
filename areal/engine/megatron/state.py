from __future__ import annotations

import os
from concurrent.futures import Future
from contextlib import nullcontext
from datetime import datetime
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu
from torch import nn
from torch_memory_saver import torch_memory_saver

from areal.api.io_struct import ParamSpec, SaveLoadMeta, WeightUpdateMeta
from areal.api.train_engine import TrainEngineStateMixin
from areal.core.dist_rollout import DistRolloutCoordinator
from areal.models.mcore.hf_load import load_weights_from_hf_with_mbridge_fast
from areal.models.mcore.hf_save import save_weights_to_hf_with_mbridge_fast
from areal.platforms import current_platform
from areal.utils import name_resolve, names
from areal.utils.constants import DIST_GROUP_DEFAULT_TIMEOUT
from areal.utils.device import clear_memory, log_gpu_stats
from areal.utils.distributed import init_custom_process_group
from areal.utils.megatron import (
    all_gather_param,
    convert_to_hf,
    get_named_parameters,
    remove_padding,
)
from areal.utils.perf_tracer import trace_perf

if TYPE_CHECKING:
    from areal.api.engine_api import InferenceEngine
    from areal.engine.megatron.protocol import MegatronEngineProtocol


class MegatronStateMixin(TrainEngineStateMixin):
    def connect_engine(
        self: MegatronEngineProtocol,
        engine: InferenceEngine,
        meta: WeightUpdateMeta,
    ):
        if self.rollout_engine is not None and self.rollout_engine != engine:
            self.logger.warning(
                f"Connected rollout engine changed from {self.rollout_engine} to {engine}."
            )
        self.rollout_engine = engine
        self.rollout_coordinator = DistRolloutCoordinator(
            rollout_engine=engine, train_engine=self
        )

        if (
            meta.type == current_platform.communication_backend
            and not self.weight_update_group_initialized
        ):
            self._init_weight_update_from_distributed(meta)
            self.weight_update_group_initialized = True

        current_platform.synchronize()
        dist.barrier(group=self.cpu_group)

    def set_version(self: MegatronEngineProtocol, version: int):
        self._version = version

    def get_version(self: MegatronEngineProtocol) -> int:
        return self._version

    def update_weights(
        self: MegatronEngineProtocol,
        meta: WeightUpdateMeta,
    ):
        self._check_rollout_engine_connected()
        if meta.type == current_platform.communication_backend:
            assert self.weight_update_group_initialized
            # In offload mode, wakes up parameters as needed to perform the update.
            tms_context = (
                torch_memory_saver.disable()
                if self.is_offload and not torch.version.hip
                else nullcontext()
            )
            with tms_context:
                self._update_weights_from_distributed(meta)
        elif meta.type == "disk":
            self._update_weights_from_disk(meta)
        else:
            raise ValueError(f"Unknown weight update type {meta.type}")

    def save(self: MegatronEngineProtocol, meta: SaveLoadMeta):
        if meta.weight_format == "hf":
            if meta.with_optim:
                raise ValueError(
                    "HF format does not support optimizer state saving, please use DCP format instead."
                )
            self._save_model_to_hf(
                meta.path,
                tokenizer=meta.tokenizer,
                processor=meta.processor,
                base_model_path=meta.base_model_path,
            )
        elif meta.weight_format == "dcp":
            self.checkpointer.save_checkpoint(meta.path, with_optimizer=meta.with_optim)
        else:
            raise ValueError(f"Unknown weight format {meta.weight_format}. ")

    def load(self: MegatronEngineProtocol, meta: SaveLoadMeta):
        if meta.weight_format == "hf":
            if meta.with_optim:
                raise ValueError(
                    "HF format does not support optimizer state loading, please use DCP format instead."
                )
            self._load_model_from_hf(meta.path)
        elif meta.weight_format == "dcp":
            self.checkpointer.load_checkpoint(meta.path, with_optimizer=meta.with_optim)
        else:
            raise ValueError(f"Unknown weight format {meta.weight_format}. ")

    def offload(self: MegatronEngineProtocol) -> None:
        """Offload model memory to CPU using torch_memory_saver.

        Ref: https://github.com/THUDM/slime/blob/main/slime/backends/megatron_utils/actor.py
        """

        log_gpu_stats("before offload model")
        clear_memory()
        torch_memory_saver.pause()

        # TODO: NCCL offload
        current_platform.synchronize()
        dist.barrier(group=self.cpu_group)
        log_gpu_stats("after offload model")

        self.is_offload = True

    def onload(self: MegatronEngineProtocol) -> None:
        """Onload model memory from CPU back to GPU using torch_memory_saver.

        Ref: https://github.com/THUDM/slime/blob/main/slime/backends/megatron_utils/actor.py
        """

        torch_memory_saver.resume()
        clear_memory()

        # TODO: NCCL onload
        current_platform.synchronize()
        dist.barrier(group=self.cpu_group)
        log_gpu_stats("after onload model")

        self.is_offload = False

    def _init_weight_update_from_distributed(
        self: MegatronEngineProtocol,
        meta: WeightUpdateMeta,
    ) -> None:
        assert meta.type == current_platform.communication_backend

        # NOTE: Processes launched with torchrun will set the following env var to True,
        # which blocks creating another TCP store for weight update.
        os.environ["TORCHELASTIC_USE_AGENT_STORE"] = str(False)
        if self.is_pipeline_parallel_head():
            assert meta.alloc_mode is not None

            fut = self.rollout_engine.init_weights_update_group(meta)

            self.logger.info(
                f"Initializing weight update group: type={meta.type} "
                f"init_method=tcp://{meta.nccl_master_address}:{meta.nccl_master_port} "
                f"group={self.weight_update_group_name}"
            )
            self.weight_update_group = init_custom_process_group(
                backend=current_platform.communication_backend,
                world_size=meta.alloc_mode.gen.world_size + 1,
                init_method=f"tcp://{meta.nccl_master_address}:{meta.nccl_master_port}",
                rank=0,
                group_name=self.weight_update_group_name,
                timeout=DIST_GROUP_DEFAULT_TIMEOUT,
            )

            fut.result()

    @trace_perf("megatron_engine.update_weights_from_distributed", category="comm")
    def _update_weights_from_distributed(
        self: MegatronEngineProtocol,
        meta: WeightUpdateMeta,
    ) -> None:
        if dist.get_rank() == 0:
            self.rollout_engine.pause_generation()

        dist.barrier(group=self.cpu_group)

        num_moe_experts = self.tf_config.num_moe_experts
        weight_chunked_mem_size = meta.weight_chunked_mem_mb * 1024 * 1024

        buffer_size = 0
        converted_named_tensors = []

        for name, param in get_named_parameters(self.model, num_moe_experts):
            if ".experts." in name:
                continue
            buffer_size = self._impl_update_weight_from_distributed(
                meta,
                name,
                param,
                converted_named_tensors,
                buffer_size,
                weight_chunked_mem_size,
            )

        # Only pipeline parallel heads CAN contain named tensors here
        if converted_named_tensors:
            self._update_bucket_weights_from_distributed(meta, converted_named_tensors)

        dist.barrier(group=self.cpu_group)

        buffer_size = 0
        named_tensors = []

        for name, param in get_named_parameters(self.model, num_moe_experts):
            if ".experts." not in name:
                continue
            buffer_size = self._impl_update_expert_weight_from_distributed(
                meta,
                name,
                param,
                named_tensors,
                buffer_size,
                weight_chunked_mem_size,
            )

        if named_tensors:
            # This function will early return if not pipeline parallel head
            self._update_bucket_expert_weights_from_distributed(meta, named_tensors)

        dist.barrier(group=self.cpu_group)

        if dist.get_rank() == 0:
            self.rollout_engine.continue_generation()

        current_platform.synchronize()
        dist.barrier(group=self.cpu_group)

    @trace_perf("megatron_engine.update_weights_from_disk", category="io")
    def _update_weights_from_disk(
        self: MegatronEngineProtocol,
        meta: WeightUpdateMeta,
    ) -> None:
        fut = Future()

        if dist.get_rank() == 0:
            fut = self.rollout_engine.update_weights_from_disk(meta)

        self._save_model_to_hf(meta.path, self.tokenizer, None)
        # dist.barrier() are called when _save_model_to_hf finished

        if dist.get_rank() == 0:
            update_name = names.update_weights_from_disk(
                self.config.experiment_name,
                self.config.trial_name,
                self.get_version(),
            )
            name_resolve.add(
                update_name, str(datetime.now().timestamp()), keepalive_ttl=120
            )

            fut.result()

        current_platform.synchronize()
        dist.barrier(group=self.cpu_group)

    def _update_bucket_weights_from_distributed(
        self: MegatronEngineProtocol,
        meta: WeightUpdateMeta,
        converted_named_tensors: list[tuple[str, nn.Parameter | torch.Tensor]],
    ) -> None:
        # Early exit when chunk size is relatively small
        if not converted_named_tensors:
            return

        self.engine_lock.acquire()

        param_specs = [
            ParamSpec(
                name=name,
                shape=tuple(tensor.shape),
                dtype=str(tensor.dtype).split("torch.")[1],
            )
            for name, tensor in converted_named_tensors
        ]

        fut = self.rollout_engine.update_weights_from_distributed(meta, param_specs)

        handles = []
        for _, param in converted_named_tensors:
            handles.append(
                dist.broadcast(
                    param.data, 0, group=self.weight_update_group, async_op=True
                )
            )
        for handle in handles:
            handle.wait()

        fut.result()

        converted_named_tensors.clear()

        self.engine_lock.release()

    def _impl_update_weight_from_distributed(
        self: MegatronEngineProtocol,
        meta: WeightUpdateMeta,
        name: str,
        param: nn.Parameter | torch.Tensor,
        converted_named_tensors: list[tuple[str, nn.Parameter | torch.Tensor]],
        buffer_size: int,
        weight_chunked_mem_size: int,
    ) -> int:
        param = all_gather_param(name, param)
        param = remove_padding(name, param, self.hf_config.vocab_size)

        if not self.is_pipeline_parallel_head():
            return buffer_size

        param_size = param.numel() * param.element_size()
        if buffer_size + param_size > weight_chunked_mem_size:
            self._update_bucket_weights_from_distributed(meta, converted_named_tensors)
            buffer_size = 0

        converted_named_tensors.extend(
            convert_to_hf(self.tf_config, self.hf_config.model_type, name, param)
        )
        buffer_size += param_size
        return buffer_size

    def _update_bucket_expert_weights_from_distributed(
        self: MegatronEngineProtocol,
        meta: WeightUpdateMeta,
        named_tensors: list[tuple[str, nn.Parameter | torch.Tensor]],
    ) -> None:
        """Gather a bucket of MoE expert weights and broadcast them.

        This function handles the distributed update for a bucket of Mixture-of-Experts
        (MoE) parameters. Since expert parameters are sharded across the expert
        parallel group, this function first performs an `all_gather` to collect all
        shards from all expert ranks.

        Once the full expert parameters are reconstructed on the pipeline parallel
        head, it converts them to the HuggingFace format and calls
        `_update_bucket_weights_from_distributed` to perform the actual broadcast
        to the inference engine.
        """

        # Early exit when chunk size is relatively small
        if not named_tensors:
            return

        group = mpu.get_expert_model_parallel_group()
        world_size = mpu.get_expert_model_parallel_world_size()

        names = [name for name, _ in named_tensors]
        all_names: list[list[str]] = [None] * world_size
        dist.all_gather_object(all_names, names, group=group)

        for rank_names in all_names:
            if len(named_tensors) != len(rank_names):
                raise RuntimeError(
                    "Named tensor count mismatch across expert parallel ranks: "
                    f"expected {len(rank_names)} but got {len(named_tensors)}"
                )

        gathered_params = [[] for _ in range(world_size)]
        handles = []
        for idx, (_, tensor) in enumerate(named_tensors):
            params = [
                torch.empty_like(tensor.data, device=current_platform.current_device())
                for _ in range(world_size)
            ]
            handle = dist.all_gather(params, tensor.data, group=group, async_op=True)
            handles.append(handle)
            for ep_rank, rank_names in enumerate(all_names):
                gathered_params[ep_rank].append((rank_names[idx], params[ep_rank]))

        for handle in handles:
            handle.wait()

        named_tensors.clear()
        if not self.is_pipeline_parallel_head():
            return

        gathered_params = sum(gathered_params, [])

        converted_hf_tensors = []
        for name, param in gathered_params:
            converted_hf_tensors.extend(
                convert_to_hf(self.tf_config, self.hf_config.model_type, name, param)
            )

        self._update_bucket_weights_from_distributed(meta, converted_hf_tensors)

    def _impl_update_expert_weight_from_distributed(
        self: MegatronEngineProtocol,
        meta: WeightUpdateMeta,
        name: str,
        param: nn.Parameter | torch.Tensor,
        named_tensors: list[tuple[str, nn.Parameter | torch.Tensor]],
        buffer_size: int,
        weight_chunked_mem_size: int,
    ) -> int:
        param = all_gather_param(name, param)
        param = remove_padding(name, param, self.hf_config.vocab_size)

        param_size = param.numel() * param.element_size()
        if (
            buffer_size + param_size
        ) * mpu.get_expert_model_parallel_world_size() > weight_chunked_mem_size:
            self._update_bucket_expert_weights_from_distributed(meta, named_tensors)
            buffer_size = 0

        named_tensors.append((name, param))
        buffer_size += param_size
        return buffer_size

    def _save_model_to_hf(
        self: MegatronEngineProtocol,
        path: str,
        tokenizer: Any | None = None,
        processor: Any | None = None,
        base_model_path: str | None = None,
    ) -> None:
        assert self.model is not None, "Model is not initialized."
        os.makedirs(path, exist_ok=True)

        save_weights_to_hf_with_mbridge_fast(
            bridge=self.bridge,
            models=self.model,
            weights_path=path,
            base_model_path=base_model_path,
            max_shard_size_byte=int(3e9),
            max_workers=None,
            is_critic=self.config.is_critic,
        )

        if dist.get_rank() == 0:
            if tokenizer is not None:
                tokenizer.save_pretrained(path)
            if processor is not None:
                processor.save_pretrained(path)

        current_platform.synchronize()
        dist.barrier(group=self.cpu_group)

    def _load_model_from_hf(self: MegatronEngineProtocol, path: str) -> None:
        assert self.model is not None, "Model is not initialized."
        load_weights_from_hf_with_mbridge_fast(
            bridge=self.bridge,
            models=self.model,
            weights_path=path,
            max_workers=None,
            is_critic=self.config.is_critic,
        )
