from __future__ import annotations

import os
from concurrent.futures import Future
from contextlib import nullcontext
from datetime import datetime
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch import nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)
from torch.distributed.tensor import DTensor
from torch_memory_saver import torch_memory_saver
from transformers import PreTrainedTokenizerFast, ProcessorMixin

from areal.api.io_struct import ParamSpec, SaveLoadMeta, WeightUpdateMeta
from areal.api.train_engine import TrainEngineStateMixin
from areal.core.dist_rollout import DistRolloutCoordinator
from areal.platforms import current_platform
from areal.utils import name_resolve, names
from areal.utils.constants import DIST_GROUP_DEFAULT_TIMEOUT
from areal.utils.device import clear_memory, log_gpu_stats
from areal.utils.distributed import init_custom_process_group
from areal.utils.fsdp import fsdp2_load_full_state_dict
from areal.utils.fsdp.checkpoint import DCPState
from areal.utils.model import is_gemma3_model, is_qwen_vl_model
from areal.utils.perf_tracer import trace_perf
from areal.utils.save_load import get_state_dict_from_repo_id_or_path

if TYPE_CHECKING:
    from collections.abc import Iterator

    from areal.api.engine_api import InferenceEngine
    from areal.engine.fsdp.protocol import FSDPEngineProtocol


class FSDPStateMixin(TrainEngineStateMixin):
    def connect_engine(
        self: FSDPEngineProtocol,
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

    def update_weights(self: FSDPEngineProtocol, meta: WeightUpdateMeta):
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

    def set_version(self: FSDPEngineProtocol, version: int):
        self._version = version

    def get_version(self: FSDPEngineProtocol) -> int:
        return self._version

    def save(self: FSDPEngineProtocol, meta: SaveLoadMeta):
        if meta.weight_format == "hf":
            self._save_model_to_hf(meta.path, meta.tokenizer, meta.processor)
        elif meta.weight_format == "dcp":
            self._save_to_dcp(meta.path, meta.with_optim)
        else:
            raise ValueError(f"Unknown weight format {meta.weight_format}. ")

        if meta.with_optim and meta.weight_format == "hf":
            self._save_optimizer_state(meta.path)

    def load(self: FSDPEngineProtocol, meta: SaveLoadMeta):
        if meta.weight_format == "hf":
            self._load_model_from_hf(meta.path)
        elif meta.weight_format == "dcp":
            self._load_from_dcp(meta.path, meta.with_optim)
        else:
            raise ValueError(f"Unknown weight format {meta.weight_format}. ")

        if meta.with_optim and meta.weight_format == "hf":
            self._load_optimizer_state(meta.path)

    def offload(self: FSDPEngineProtocol) -> None:
        """Offload model memory to CPU using torch_memory_saver.

        Ref: https://github.com/THUDM/slime/blob/main/slime/backends/fsdp_utils/actor.py
        """

        log_gpu_stats("before offload model")

        # Use torch_memory_saver to pause CUDA memory
        clear_memory()
        torch_memory_saver.pause()

        current_platform.synchronize()
        dist.barrier(group=self.cpu_group)
        log_gpu_stats("after offload model")

        self.is_offload = True

    def onload(self: FSDPEngineProtocol) -> None:
        """Onload model memory from CPU back to GPU using torch_memory_saver.

        Ref: https://github.com/THUDM/slime/blob/main/slime/backends/fsdp_utils/actor.py
        """

        torch_memory_saver.resume()

        current_platform.synchronize()
        dist.barrier(group=self.cpu_group)
        log_gpu_stats("after onload model")

        self.is_offload = False

    def _get_model_name_parameters(
        self: FSDPEngineProtocol,
    ) -> Iterator[tuple[str, nn.Parameter]]:
        name_params_iterator = self.model.named_parameters()
        if self.is_vision_model and is_qwen_vl_model(self.model_config.model_type):
            for name, value in name_params_iterator:
                new_name = name.replace("model.", "", 1).replace(
                    "language_model", "model"
                )
                yield new_name, value
        elif self.is_vision_model and is_gemma3_model(self.model_config.model_type):
            for name, value in name_params_iterator:
                new_name = name.replace("model.", "", 1)
                if new_name.startswith("language_model."):
                    new_name = new_name.replace(
                        "language_model.", "language_model.model.", 1
                    )
                elif new_name.startswith("lm_head."):
                    new_name = new_name.replace(
                        "lm_head.", "language_model.lm_head.", 1
                    )
                yield new_name, value
        else:
            yield from name_params_iterator

    def _get_full_tensor(self: FSDPEngineProtocol, param: nn.Parameter) -> torch.Tensor:
        """Get full tensor from a parameter, handling DTensor and CPU offloaded tensors."""
        tensor = param.data
        if isinstance(tensor, DTensor):
            # For non-offloaded DTensor, directly call full_tensor()
            if tensor.device.type != "cpu":
                return tensor.full_tensor()

            # Handle CPU offloaded DTensor: reconstruct DTensor from local tensor
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

    def _update_bucket_weights_from_distributed(
        self: FSDPEngineProtocol,
        meta: WeightUpdateMeta,
        named_tensors: list[tuple[str, nn.Parameter | torch.Tensor]],
    ):
        # Early exit when chunk size is relatively small
        if not named_tensors:
            return

        param_specs = [
            ParamSpec(
                name=name,
                shape=tuple(tensor.shape),
                dtype=str(tensor.dtype).split("torch.")[1],
            )
            for name, tensor in named_tensors
        ]

        fut = self.rollout_engine.update_weights_from_distributed(meta, param_specs)

        handles = []
        for _, tensor in named_tensors:
            handles.append(
                dist.broadcast(
                    tensor, src=0, group=self.weight_update_group, async_op=True
                )
            )
        for handle in handles:
            handle.wait()

        fut.result()

        named_tensors.clear()

    def _init_weight_update_from_distributed(
        self: FSDPEngineProtocol, meta: WeightUpdateMeta
    ):
        assert meta.type == current_platform.communication_backend

        # NOTE: Processes launched with torchrun will set the following env var to True,
        # which blocks creating another TCP store for weight update.
        os.environ["TORCHELASTIC_USE_AGENT_STORE"] = str(False)
        if dist.get_rank() == 0:
            assert meta.alloc_mode is not None

            fut = self.rollout_engine.init_weights_update_group(meta)

            self.logger.info(
                f"Initializing weight update group: type={meta.type} "
                f"init_method=tcp://{meta.nccl_master_address}:{meta.nccl_master_port} "
                f"group={meta.nccl_group_name}"
            )
            self.weight_update_group = init_custom_process_group(
                backend=current_platform.communication_backend,
                world_size=meta.alloc_mode.gen.world_size + 1,
                init_method=f"tcp://{meta.nccl_master_address}:{meta.nccl_master_port}",
                rank=0,
                group_name=meta.nccl_group_name,
                timeout=DIST_GROUP_DEFAULT_TIMEOUT,
            )

            fut.result()

    @trace_perf("fsdp_engine.update_weights_from_distributed", category="comm")
    def _update_weights_from_distributed(
        self: FSDPEngineProtocol, meta: WeightUpdateMeta
    ):
        """Broadcast parameters (chunked) from rank 0 (FSDP2 compatible)."""

        if dist.get_rank() == 0:
            self.rollout_engine.pause_generation()

        dist.barrier(group=self.cpu_group)

        weight_chunked_mem_size = meta.weight_chunked_mem_mb * 1024 * 1024
        main_rank = dist.get_rank() == 0

        buffer_size = 0
        named_tensors: list[tuple[str, torch.Tensor]] = []

        for name, param in self._get_model_name_parameters():
            tensor = self._get_full_tensor(param)

            # Ranks other than 0 only help to get the full tensor
            if not main_rank:
                continue

            tensor_size = tensor.numel() * tensor.element_size()

            if tensor_size + buffer_size > weight_chunked_mem_size:
                self._update_bucket_weights_from_distributed(meta, named_tensors)
                buffer_size = 0

            named_tensors.append((name, tensor))
            buffer_size += tensor_size

        # Process remaining parameters
        if named_tensors:
            self._update_bucket_weights_from_distributed(meta, named_tensors)

        dist.barrier(group=self.cpu_group)

        if dist.get_rank() == 0:
            self.rollout_engine.continue_generation()

        current_platform.synchronize()
        dist.barrier(group=self.cpu_group)

    @trace_perf("fsdp_engine.update_weights_from_disk", category="io")
    def _update_weights_from_disk(self: FSDPEngineProtocol, meta: WeightUpdateMeta):
        fut = Future()

        if dist.get_rank() == 0:
            fut = self.rollout_engine.update_weights_from_disk(meta)

        assert meta.path is not None
        self._save_model_to_hf(meta.path, self.tokenizer, self.processor)
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

    def _save_model_to_hf(
        self: FSDPEngineProtocol,
        path: str,
        tokenizer: PreTrainedTokenizerFast | None,
        processor: ProcessorMixin | None,
    ):
        """Save model in HuggingFace format."""
        if self.model is None:
            raise RuntimeError("Model not initialized")
        os.makedirs(path, exist_ok=True)

        # FSDP2 checkpoint saving
        # Get full state dict with FSDP2
        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        state_dict = get_model_state_dict(self.model, options=options)

        # save huggingface model on rank 0
        if dist.get_rank() == 0:
            os.makedirs(path, exist_ok=True)
            self.model.save_pretrained(path, state_dict=state_dict)
            self.model_config.save_pretrained(path)
            if tokenizer is not None:
                tokenizer.save_pretrained(path)
            if processor is not None:
                processor.save_pretrained(path)
        dist.barrier(group=self.cpu_group)

    def _load_model_from_hf(self: FSDPEngineProtocol, path: str):
        """Load model from HuggingFace format."""
        if dist.get_rank() == 0:
            full_state = get_state_dict_from_repo_id_or_path(path)
        else:
            full_state = {}

        fsdp2_load_full_state_dict(
            self.model,
            full_state,
            self.cpu_offload,
            tie_word_embeddings=self.model_config.tie_word_embeddings,
        )

    def _save_to_dcp(
        self: FSDPEngineProtocol,
        path: str,
        with_optim: bool,
    ):
        """Save model in PyTorch Distributed Checkpoint (DCP) format."""
        if self.model is None:
            raise RuntimeError("Model not initialized")

        os.makedirs(path, exist_ok=True)

        dcp_state = DCPState(self.model, self.optimizer if with_optim else None)
        state_dict = {"dcp": dcp_state}
        dcp.save(state_dict, checkpoint_id=path)

    def _load_from_dcp(self: FSDPEngineProtocol, path: str, with_optim: bool):
        """Load model from Distributed Checkpoint (DCP) format."""
        if self.model is None:
            raise RuntimeError("Model not initialized")

        dcp_state = DCPState(self.model, self.optimizer if with_optim else None)
        state_dict = {"dcp": dcp_state}
        dcp.load(
            state_dict=state_dict,
            checkpoint_id=path,
        )

    def _save_optimizer_state(self: FSDPEngineProtocol, path: str):
        assert self.optimizer is not None
        assert dist.is_initialized()
        rank = dist.get_rank()
        shard_path = os.path.join(
            path, f"optim_world_size_{self.world_size}_rank_{rank}.pt"
        )
        state_dict = self.optimizer.state_dict()
        torch.save(state_dict, shard_path)
        dist.barrier(group=self.cpu_group)

    def _load_optimizer_state(self: FSDPEngineProtocol, path: str):
        assert self.optimizer is not None
        assert dist.is_initialized()
        rank = dist.get_rank()
        shard_path = os.path.join(
            path, f"optim_world_size_{self.world_size}_rank_{rank}.pt"
        )
        optimizer_state_dict = torch.load(shard_path, weights_only=False)
        self.optimizer.load_state_dict(optimizer_state_dict)
        dist.barrier(group=self.cpu_group)
