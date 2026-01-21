# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import contextlib
import dataclasses
import functools
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed
import torch.nn as nn
import torch.utils.checkpoint
import transformers

from realhf.api.core import model_api
from realhf.api.core.config import ModelName
from realhf.base import constants, logging, topology
from realhf.impl.model.comm.global_comm import NCCLProcessGroupInfo
from realhf.impl.model.comm.param_realloc import (
    ReparallelizeReceiverStep,
    ReparallelizeSenderStep,
    ReparallelizeTraget,
    _derive_reparallelize_comm_plan,
    is_trainable,
)
from realhf.impl.model.nn.flatten_param import set_intervals, slice_intervals
from realhf.impl.model.utils.padding import pad_input, unpad_input

from .flatten_param import build_param_spec, map_param_to_contigous_memory
from .real_llm_base import (
    OutputHead,
    ParallelActorHead,
    PipeCacheData,
    PipeTransferData,
    ReaLModelBlock,
    SequenceParallelCriticHead,
    VocabPositionEmbedding,
)
from .real_llm_generate import generate
from .real_llm_parallel import partition_pipeline_layers

logger = logging.getLogger("ReaLModel Interface")


def chunked_bcast(x, src, group, chunk_size_bytes=1024 * 1024**2):
    assert len(x.shape) == 1
    n_chunks = (x.numel() * x.dtype.itemsize + chunk_size_bytes - 1) // chunk_size_bytes
    chunk_size = chunk_size_bytes // x.dtype.itemsize
    for i in range(n_chunks):
        if isinstance(x, torch.Tensor):
            torch.distributed.broadcast(
                x[i * chunk_size : (i + 1) * chunk_size], src=src, group=group
            )
        else:
            assert isinstance(x, list) and len(x) == n_chunks
            torch.distributed.broadcast(x[i], src=src, group=group)


@dataclasses.dataclass
class DuckModelOutput:
    logits: Optional[Union[List[torch.Tensor], torch.Tensor]] = None


@dataclasses.dataclass
class DuckGenerationOutput:
    sequences: torch.Tensor
    scores: Optional[torch.Tensor] = None
    logits_mask: Optional[torch.Tensor] = None


def _sync_embedding_and_output_weights(layers: nn.ModuleList):
    pp_size = constants.pipe_parallel_world_size()
    pp_rank = constants.pipe_parallel_rank()
    if pp_size == 1:
        old_head_w = layers[-1].weight.data
        layers[-1].weight = layers[0].wte.weight
        del old_head_w
        layers[0].wte.weight.zero_out_wgrad = True
        return

    if pp_rank != 0 and pp_rank != pp_size - 1:
        return

    if pp_rank == 0:
        weight = layers[0].wte.weight
        weight.shared_embedding = True
    else:
        weight = layers[-1].weight
        weight.data.fill_(0.0)
        # To make Megatron happy
        weight.shared = True
        weight.shared_embedding = True

    group = constants.grid().embedding_proc_group
    torch.distributed.all_reduce(weight.data, group=group)


class ReaLModel(nn.Module):
    """The transformer model used in ReaL.

    This model supports 3D parallelism, offloaded inference,
    and parameter reallocation. It is usually more efficient
    than HuggingFace implementations.

    During construction, model parameters are not instantiated
    immediately because the model may be redistributed.
    The method ``instantiate`` should be called before using
    model parameters, e.g., forward or state dict.

    :param config: The model configuration.
    :type config: model_api.ReaLModelConfig
    :param dtype: The data type of the model.
    :type dtype: Optional[torch.dtype], optional
    :param device: The device of the model.
    """

    def __init__(
        self,
        config: model_api.ReaLModelConfig,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
        hf_model_family: Optional[str] = None,
    ):
        super().__init__()
        if dtype is None:
            dtype = torch.float16
        self.config = config
        self.dtype = dtype
        self.device = device

        # The main attribute of the model: layers,
        # including the embedding layer, decoder layers, and the output head.
        self.layer_mapping = partition_pipeline_layers(
            config,
            constants.pipe_parallel_world_size(),
        )
        self.layer_idx_start = self.layer_mapping[constants.pipe_parallel_rank()][0]
        self.layer_idx_end = self.layer_mapping[constants.pipe_parallel_rank()][1]
        self.num_stages = constants.pipe_parallel_world_size()

        self.layers = nn.ModuleList()

        # The model is lazily instantiated due to parameter reallocation.
        # For models that will be redistributed, we instantiate replica 0
        # and do not instantiate other replicas.
        self._instantiated = False
        self._instantiation_hooks = []

        # Attributes used for parameter reallocation.
        self._reparallelize_targets: Dict[
            Tuple[ModelName, ModelName], ReparallelizeTraget
        ] = {}

        # Attributes used for offload.
        self._offload_buffer = None
        self._offloaded = False

        # Attributes used for flattening parameters.
        self.head_param_point_to_embedding = (
            self.config.tied_embedding
            and not self.config.is_critic
            and constants.pipe_parallel_world_size() == 1
        )
        self._param_spec, self._param_size = build_param_spec(
            list(range(self.layer_idx_start, self.layer_idx_end)),
            self.config,
            tp_size=constants.tensor_parallel_world_size(),
            pp_size=constants.pipe_parallel_world_size(),
            dp_size=constants.data_parallel_world_size(),
            head_param_point_to_embedding=self.head_param_point_to_embedding,
        )
        self.contiguous_param = None

        self.hf_model_family = hf_model_family

    def save_to_hf(self, tokenizer, save_dir):
        return getattr(self, f"to_{self.hf_model_family}")(tokenizer, save_dir)

    def load_from_hf(self, load_dir, init_critic_from_actor):
        return getattr(self, f"from_{self.hf_model_family}")(
            load_dir, init_critic_from_actor
        )

    @property
    def pre_process(self):
        # A workaround to make Megatron-LM backend happy.
        if constants.pipe_parallel_rank() == 0:
            return self.layers[0]
        elif constants.pipe_parallel_rank() == constants.pipe_parallel_world_size() - 1:
            return self.layers[-1]
        return None

    @property
    def post_process(self):
        # A workaround to make Megatron-LM backend happy.
        if constants.pipe_parallel_rank() == constants.pipe_parallel_world_size() - 1:
            return self.layers[-1]
        return None

    def shared_embedding_or_output_weight(self) -> None | torch.Tensor:
        # NOTE: Use this name in consistent with Megatron-LM.
        if not self.config.tied_embedding or self.config.is_critic:
            return None
        if constants.is_first_pipe_stage():
            return self.layers[0].wte.weight
        elif constants.is_last_pipe_stage():
            return self.layers[-1].weight
        return None

    def instantiate(self):
        """Instantiate the model parameters.

        Note that users can append hooks to this method to do more
        processing, such as loading from HuggingFace models.
        """
        assert not self._instantiated or self.contiguous_param.numel() == 0
        layers = []
        for idx in range(self.layer_idx_start, self.layer_idx_end):
            layers.append(self._build_layer(idx, self.config))
        self.layers = nn.ModuleList(layers)

        if self.config.tied_embedding and not self.config.is_critic:
            _sync_embedding_and_output_weights(self.layers)

        self.contiguous_param = torch.empty(
            self._param_size, dtype=self.dtype, device=self.device
        )
        map_param_to_contigous_memory(
            self.layers,
            self.config,
            self.head_param_point_to_embedding,
            self._param_spec,
            self.contiguous_param,
            self.layer_idx_start,
            allocate_only=False,
        )

        for h in self._instantiation_hooks:
            h()

        self._instantiated = True
        self._instantiation_hooks = []

    @property
    def num_layers(self):
        """Return the number of embedding or transformer layers in this
        pipeline stage."""
        return self.layer_idx_end - self.layer_idx_start

    @property
    def is_critic(self):
        return self.config.is_critic

    def _build_layer(self, idx: int, config: model_api.ReaLModelConfig) -> nn.Module:
        dtype = self.dtype
        device = self.device
        if idx == 0:
            l = VocabPositionEmbedding(config, dtype=dtype, device=device)
        elif idx == config.n_layers + 1:
            l = self._build_output_head(config)
        else:
            l = ReaLModelBlock(
                config=config,
                layer_index=idx - 1,
                output_layernorm=(idx == config.n_layers),
                dtype=dtype,
                device=device,
            )
        return l

    def _build_output_head(self, config: model_api.ReaLModelConfig) -> nn.Module:
        dtype = self.dtype
        device = self.device
        if config.is_critic and constants.sequence_parallel():
            l = SequenceParallelCriticHead(
                config.hidden_dim,
                1,
                bias=False,
                device=device,
                dtype=dtype,
            )
        elif not config.is_critic and constants.tensor_parallel_world_size() > 1:
            l = ParallelActorHead(
                config.hidden_dim,
                config.vocab_size,
                norm_head=self.config.norm_head,
                norm_softmax=self.config.norm_softmax,
                bias=False,
                gradient_accumulation_fusion=constants.gradient_accumulation_fusion(),
                device=device,
                dtype=dtype,
            )
        else:
            l = OutputHead(
                config.hidden_dim,
                1 if config.is_critic else config.vocab_size,
                bias=False,
                norm_head=self.config.norm_head,
                norm_softmax=self.config.norm_softmax,
                device=device,
                dtype=dtype,
            )
        return l

    def async_offload(self):
        """Trigger offload asynchronously."""
        if not constants.use_cuda():
            return
        assert not self._offloaded
        assert self._instantiated
        if self._offload_buffer is None:
            self._offload_buffer = torch.empty_like(
                self.contiguous_param,
                dtype=self.dtype,
                device="cpu",
                pin_memory=True,
            )
        else:
            assert self._offload_buffer.shape == self.contiguous_param.shape
        dummy_tensor = torch.tensor((), device=self.device, dtype=self.dtype)
        self._offload_stream = torch.cuda.Stream()
        self._offload_event = torch.cuda.Event()
        self.contiguous_param = None
        for i, l in enumerate(self.layers):
            layer_idx = self.layer_idx_start + i
            with torch.cuda.stream(self._offload_stream):
                for k, p in l.named_parameters():
                    spec = self._param_spec[f"{layer_idx}.{k}"]
                    if (
                        self.head_param_point_to_embedding
                        and layer_idx == self.config.n_layers + 1
                    ):
                        continue
                    self._offload_buffer[spec.start_idx : spec.end_idx].copy_(
                        p.data.view(-1), non_blocking=True
                    )
                    p.data = dummy_tensor
        self._offload_event.record(self._offload_stream)
        self._offloaded = True

    def wait_for_offload(self):
        """Wait for offload to finish."""
        if not constants.use_cuda():
            return
        assert self._offloaded
        torch.cuda.current_stream().wait_event(self._offload_event)

    def __overlapped_load_forward(
        self, x: PipeTransferData, ys: List[PipeCacheData]
    ) -> Tuple[PipeTransferData, List[PipeCacheData]]:
        assert len(ys) == self.num_layers
        raw_pp_input = x.pp_input
        self.contiguous_param = torch.empty(
            self._param_size, dtype=self.dtype, device=self.device
        )
        map_param_to_contigous_memory(
            self.layers,
            self.config,
            self.head_param_point_to_embedding,
            self._param_spec,
            self.contiguous_param,
            self.layer_idx_start,
            allocate_only=True,
        )
        self.wait_for_offload()

        stream = torch.cuda.Stream()
        events: List[torch.cuda.Event] = [
            torch.cuda.Event() for _ in range(self.num_layers)
        ]
        with torch.cuda.stream(stream):
            for layer_idx, y, l, e in zip(
                range(self.layer_idx_start, self.layer_idx_end),
                ys,
                self.layers,
                events,
            ):
                # NOTE: although we can do more fine-grained overlapping, the overhead that can be
                # reduced is very small (~50ms), which is unnecessary for now.
                for k, v in l.named_parameters():
                    spec = self._param_spec[f"{layer_idx}.{k}"]
                    v.data.copy_(
                        self._offload_buffer[spec.start_idx : spec.end_idx].view(
                            spec.shape
                        ),
                        non_blocking=True,
                    )
                e: torch.cuda.Event
                e.record(stream)

        for layer_idx, y, l, e in zip(
            range(self.layer_idx_start, self.layer_idx_end),
            ys,
            self.layers,
            events,
        ):
            torch.cuda.default_stream().wait_event(e)
            x = l(x, y)
            x.pp_input = x.pp_output
        self._offloaded = False
        x.pp_input = raw_pp_input
        return x, ys

    def __forward(
        self, x: PipeTransferData, ys: List[PipeCacheData]
    ) -> Tuple[PipeTransferData, List[PipeCacheData]]:
        layers = self.layers
        assert len(ys) == len(layers), (len(ys), len(layers))
        raw_pp_input = x.pp_input
        for i, (layer, y) in enumerate(zip(layers, ys)):
            x = layer(x, y)
            x.pp_input = x.pp_output
        # Finally, pp_input is the input of this pipeline stage (maybe across several layers),
        # pp_output is the output of this pipeline stage.
        # In the first stage, pp_input is None.
        x.pp_input = raw_pp_input
        return x, ys

    def forward(
        self, x: PipeTransferData, ys: List[PipeCacheData]
    ) -> Tuple[PipeTransferData, List[PipeCacheData]]:
        if x.max_seqlen is not None and not isinstance(x.max_seqlen, int):
            x.max_seqlen = int(x.max_seqlen)
        if x.cu_seqlens is not None and not isinstance(x.cu_seqlens, torch.IntTensor):
            x.cu_seqlens = x.cu_seqlens.int()

        # Copy input tensor to a pinned buffer.
        tp_size = constants.tensor_parallel_world_size()
        batch_length = None
        if ys[0].packed_input_ids is not None:
            batch_length = ys[0].packed_input_ids.shape[0]
        if x.pp_input is not None:
            batch_length = x.pp_input.shape[0]
        assert batch_length is not None
        padded_batch_length = (batch_length + tp_size - 1) // tp_size * tp_size
        pad_size = padded_batch_length - batch_length

        if (
            constants.sequence_parallel()
            and pad_size > 0
            and ys[0].packed_input_ids is not None
        ):
            _cu_seqlens = x.cu_seqlens
            _max_seqlen = x.max_seqlen
            _input_ids = ys[0].packed_input_ids
            _pp_input = x.pp_input

            x.cu_seqlens = torch.nn.functional.pad(
                x.cu_seqlens, (0, 1), value=padded_batch_length
            )
            x.max_seqlen = max(x.max_seqlen, padded_batch_length - batch_length)
            if ys[0].packed_input_ids is not None:
                input_ids_buf = torch.zeros(
                    (padded_batch_length,),
                    dtype=torch.long,
                    device=self.device,
                )
                input_ids_buf[:batch_length] = ys[0].packed_input_ids
                ys[0].packed_input_ids = input_ids_buf

            if x.pp_input is not None:
                pp_input_buf = torch.zeros(
                    (padded_batch_length, *x.pp_input.shape[1:]),
                    dtype=x.pp_input.dtype,
                    device=self.device,
                )
                pp_input_buf[:batch_length] = x.pp_input
                x.pp_input = pp_input_buf

        # Main forward calls.
        if not self._offloaded:
            x, ys = self.__forward(x, ys)
        else:
            x, ys = self.__overlapped_load_forward(x, ys)

        # Resume from padding.
        if (
            constants.sequence_parallel()
            and pad_size > 0
            and ys[0].packed_input_ids is not None
        ):
            x.pp_output = x.pp_output[:-pad_size]

            x.pp_input = _pp_input
            ys[0].packed_input_ids = _input_ids
            x.cu_seqlens = _cu_seqlens
            x.max_seqlen = _max_seqlen

            if x.store_kv_cache:
                for y in ys:
                    if y.k_cache is not None:
                        y.k_cache = y.k_cache[:-pad_size]
                    if y.v_cache is not None:
                        y.v_cache = y.v_cache[:-pad_size]

        # Release the memory used for TP gathering.
        constants.clear_global_memory_buffer()
        return x, ys

    def _forward(
        self,
        input_ids: torch.LongTensor,
        cu_seqlens: torch.IntTensor,
        position_ids: torch.LongTensor,
        hidden_states: Optional[torch.Tensor],
        k_caches: Optional[List[torch.Tensor]],
        v_caches: Optional[List[torch.Tensor]],
        cache_seqlens: Optional[torch.IntTensor],
        max_seqlen: Optional[int],
    ):
        if k_caches is None:
            assert v_caches is None
            assert cache_seqlens is None
            k_caches = [None] * self.num_layers
            v_caches = [None] * self.num_layers

        h = hidden_states
        for idx, l in enumerate(self.layers):
            if isinstance(l, VocabPositionEmbedding):
                h = l._forward(input_ids, position_ids)
            elif isinstance(l, ReaLModelBlock):
                h, _, _ = l._forward(
                    h,
                    cu_seqlens=cu_seqlens,
                    k_cache=k_caches[idx],
                    v_cache=v_caches[idx],
                    cache_seqlens=cache_seqlens,
                    max_seqlen=max_seqlen,
                )
            elif isinstance(
                l,
                (
                    OutputHead,
                    SequenceParallelCriticHead,
                    ParallelActorHead,
                ),
            ):
                h = l._forward(h)
            else:
                raise NotImplementedError(f"Unsupported layer type {type(l)}")

        return h

    def state_dict(self, *args, **kwargs):
        """Map layer indices to global layer indices."""
        state_dict = self.layers.state_dict(*args, **kwargs)
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.lstrip("module.").lstrip("layers.")
            local_idx = int(k.split(".")[0])
            name = k.split(".", 1)[1]
            new_state_dict[f"{local_idx + self.layer_idx_start}.{name}"] = v
        return new_state_dict

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.split(".", 1)[1]
            global_idx = int(k.split(".")[0])
            new_state_dict[f"layers.{global_idx - self.layer_idx_start}.{name}"] = v
        return super().load_state_dict(
            new_state_dict,
            strict=strict,
            assign=assign,
        )

    def build_reparallelization_plan(
        self,
        from_model_name: ModelName,
        to_model_name: ModelName,
        from_topo: topology.ProcessTopology,
        to_topo: topology.ProcessTopology,
        to_model_config: model_api.ReaLModelConfig,
        pg_info: NCCLProcessGroupInfo,
        from_model_config: None | model_api.ReaLModelConfig = None,
    ):
        if from_model_config is None:
            from_model_config = self.config
        to_layer_mapping = partition_pipeline_layers(
            to_model_config,
            to_topo.get_dim("pipe"),
        )
        to_layers_handle_dict = {}
        to_layer_indices = []
        if constants.has_model_name(to_model_name):
            with constants.model_scope(to_model_name):
                to_pp_rank = constants.pipe_parallel_rank()
                to_layer_indices = list(
                    range(
                        to_layer_mapping[to_pp_rank][0],
                        to_layer_mapping[to_pp_rank][1],
                    )
                )
                for _to_layer_idx in to_layer_indices:
                    l = self._build_layer(_to_layer_idx, to_model_config)
                    for v in l.parameters():
                        v.data = torch.tensor((), dtype=self.dtype, device=self.device)
                    to_layers_handle_dict[_to_layer_idx] = l
        to_model_head_param_point_to_embedding = (
            to_model_config.tied_embedding
            and not to_model_config.is_critic
            and to_topo.get_dim("pipe") == 1
        )
        to_param_spec, to_param_size = build_param_spec(
            to_layer_indices,
            to_model_config,
            tp_size=to_topo.get_dim("tensor"),
            dp_size=to_topo.get_dim("data"),
            pp_size=to_topo.get_dim("pipe"),
            head_param_point_to_embedding=to_model_head_param_point_to_embedding,
        )
        if len(to_layer_indices) > 0:
            to_layer_idx_start = min(to_layer_indices)
            to_layer_idx_end = max(to_layer_indices) + 1
        else:
            to_layer_idx_start = to_layer_idx_end = -1
        to_layers_handle = nn.ModuleList(
            [to_layers_handle_dict[i] for i in to_layer_indices]
        )

        comm_plan = _derive_reparallelize_comm_plan(
            from_model_name=from_model_name,
            to_model_name=to_model_name,
            from_topo=from_topo,
            to_topo=to_topo,
            from_model_config=from_model_config,
            to_model_config=to_model_config,
            pg_info=pg_info,
            dtype=self.dtype,
        )
        rtgt = ReparallelizeTraget(
            comm_plan=comm_plan,
            to_param_spec=to_param_spec,
            to_param_size=to_param_size,
            to_layers_handle=to_layers_handle,
            to_layer_start_idx=to_layer_idx_start,
            to_layer_end_idx=to_layer_idx_end,
        )
        self._reparallelize_targets[(from_model_name, to_model_name)] = rtgt

    # FIXME: we can get topo given model name from constants
    def build_reparallelized_layers_async(
        self,
        from_model_name: ModelName,
        to_model_name: ModelName,
        from_topo: topology.ProcessTopology,
        to_topo: topology.ProcessTopology,
        to_model_config: model_api.ReaLModelConfig,
        pg_info: NCCLProcessGroupInfo,
    ) -> Tuple[nn.ModuleList, torch.Tensor, torch.Tensor]:
        """Trigger the parameter realloaction from the source model to the
        target model."""

        assert not (is_trainable(from_model_name) and is_trainable(to_model_name))
        assert is_trainable(from_model_name) or is_trainable(to_model_name)

        if (from_model_name, to_model_name) not in self._reparallelize_targets:
            self.build_reparallelization_plan(
                from_model_name,
                to_model_name,
                from_topo,
                to_topo,
                to_model_config,
                pg_info,
            )
        rtgt = self._reparallelize_targets[(from_model_name, to_model_name)]

        # Since the default implementation of PyTorch optimizers holds
        # the reference of trainable parameters, we cannot deallocate
        # them even after parameter reallocation. Therefore, there is no
        # need to release and re-allocate the trainable parameters back-and-forth.
        # We simply store the layer handles and fetch them when converting back.
        with constants.model_scope(from_model_name):
            from_model_ranks = constants.parallelism_group_ranks()
        if not is_trainable(from_model_name):
            if torch.distributed.get_rank() in from_model_ranks:
                dummy_tensor = torch.tensor((), dtype=self.dtype, device=self.device)
                for p in self.layers.parameters():
                    p.data = dummy_tensor
                self.contiguous_param = dummy_tensor
            return None, None, 0.0

        # The following tensor holds the contiguous memory of incoming parameters
        # If this process is not a receiver, to_param_size is 0 and it's an empty tensor.
        to_contiguous_param = torch.zeros(
            rtgt.to_param_size,
            dtype=self.dtype,
            device=constants.current_device(),
        )
        to_model_head_param_point_to_embedding = (
            to_model_config.tied_embedding
            and not to_model_config.is_critic
            and to_topo.get_dim("pipe") == 1
        )
        map_param_to_contigous_memory(
            rtgt.to_layers_handle,
            to_model_config,
            to_model_head_param_point_to_embedding,
            rtgt.to_param_spec,
            to_contiguous_param,
            rtgt.to_layer_start_idx,
            allocate_only=True,
        )

        # Allocate tensors in advance to reduce overhead.
        recv_buf_specs = []
        send_buf_specs = []
        comm_volume = torch.zeros(
            (), dtype=torch.long, device=constants.current_device()
        )
        for step in rtgt.comm_plan:
            if (
                isinstance(step, ReparallelizeReceiverStep)
                and step.rank == torch.distributed.get_rank()
            ):
                if step.rank == step.src:
                    # TODO: we can develop a kernel to directly move the
                    # memory without creating an intermediate buffer.
                    buf = slice_intervals(
                        self.contiguous_param,
                        step.sender_param_intervals_cpu,
                        intervals_cuda=step.sender_param_intervals_cuda,
                        max_interval_size=step.sender_max_interval_size,
                        output_size=step.param_size,
                    )
                else:
                    buf = torch.zeros(
                        step.param_size,
                        dtype=step.param_dtype,
                        device=constants.current_device(),
                    )
                    comm_volume += buf.numel()

                recv_buf_specs.append(
                    dict(
                        src=buf,
                        dst=to_contiguous_param,
                        intervals_cpu=step.receiver_param_intervals_cpu,
                        intervals_cuda=step.receiver_param_intervals_cuda,
                        max_interval_size=step.receiver_max_interval_size,
                    )
                )

            if (
                isinstance(step, ReparallelizeSenderStep)
                and step.rank == torch.distributed.get_rank()
            ):
                if step.group is not None:
                    buf = slice_intervals(
                        self.contiguous_param,
                        step.param_intervals_cpu,
                        intervals_cuda=step.param_intervals_cuda,
                        max_interval_size=step.max_interval_size,
                        output_size=step.param_size,
                    )
                    send_buf_specs.append(buf)

        # Run boradcast!
        recv_buf_cnt = 0
        recv_events = []
        for step in rtgt.comm_plan:
            if constants.use_cuda():
                s = torch.cuda.Stream()
                ctx = torch.cuda.stream(s)
            else:
                ctx = contextlib.nullcontext()
            with ctx:
                if (
                    isinstance(step, ReparallelizeReceiverStep)
                    and step.rank == torch.distributed.get_rank()
                ):
                    if constants.use_cuda():
                        e = torch.cuda.Event()
                    else:
                        e = None
                    if step.rank != step.src:
                        buf = recv_buf_specs[recv_buf_cnt]["src"]
                        chunked_bcast(buf, src=step.src, group=step.group)
                    if constants.use_cuda():
                        e.record(s)
                    recv_events.append(e)
                    recv_buf_cnt += 1

                if (
                    isinstance(step, ReparallelizeSenderStep)
                    and step.rank == torch.distributed.get_rank()
                ):
                    if step.group is not None:
                        buf = send_buf_specs.pop(0)
                        chunked_bcast(buf, src=step.rank, group=step.group)

        # Post-processing.
        assert len(send_buf_specs) == 0, len(send_buf_specs)
        assert recv_buf_cnt == len(recv_buf_specs), (
            len(recv_buf_specs),
            recv_buf_cnt,
        )
        # assert len(state_dict) == 0
        assert len(recv_events) == len(recv_buf_specs)
        for e, x in zip(recv_events, recv_buf_specs):
            if constants.use_cuda():
                torch.cuda.current_stream().wait_event(e)
            set_intervals(**x)

        return rtgt.to_layers_handle, to_contiguous_param, comm_volume

    def patch_reparallelization(self, x, eta):
        if eta == 1.0:
            self.layers, self.contiguous_param = x
        else:
            new_layers, new_param = x
            self.contiguous_param = eta * new_param + (1 - eta) * self.contiguous_param
            map_param_to_contigous_memory(
                self.layers,
                self.config,
                self.head_param_point_to_embedding,
                param_spec=self._param_spec,
                contiguous_param=self.contiguous_param,
                layer_idx_offset=self.layer_idx_start,
                allocate_only=False,
            )
            dummy_tensor = torch.tensor((), dtype=self.dtype, device=self.device)
            for p in new_layers.parameters():
                p.data = dummy_tensor
        assert self.layers is not None
        assert self.contiguous_param is not None
        assert self.contiguous_param.shape[0] > 0
        for l in self.layers:
            for p in l.parameters():
                p.requires_grad_()


# a helper function to make real_model look like huggingface model
def generate_helper(
    self: ReaLModel,
    tokenizer: transformers.PreTrainedTokenizerFast,
    input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    packed_input_ids: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    gconfig: model_api.GenerationHyperparameters = dataclasses.field(
        default_factory=model_api.GenerationHyperparameters
    ),
) -> DuckGenerationOutput:
    assert (packed_input_ids is None) == (cu_seqlens is None) == (max_seqlen is None)
    if attention_mask is None and input_ids is not None:
        attention_mask = torch.ones_like(input_ids)
    if packed_input_ids is None and attention_mask is not None:
        packed_input_ids, _, cu_seqlens, max_seqlen = unpad_input(
            input_ids, attention_mask
        )
    current_forward = self.forward
    self.forward = functools.partial(ReaLModel.forward, self)
    seq, scores, mask, _, _ = generate(
        model=self,
        tokenizer=tokenizer,
        packed_input_ids=packed_input_ids,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        gconfig=gconfig,
    )
    self.forward = current_forward
    return DuckGenerationOutput(seq, scores, mask)


# a helper function to make real_model look like huggingface model
def forward_helper(
    self: ReaLModel,
    input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    packed_input_ids: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
) -> DuckModelOutput:
    assert (packed_input_ids is None) == (cu_seqlens is None) == (max_seqlen is None)
    build_packed = False
    if attention_mask is None and input_ids is not None:
        attention_mask = torch.ones_like(input_ids)
    if packed_input_ids is None and attention_mask is not None:
        build_packed = True
        packed_input_ids, indices, cu_seqlens, max_seqlen = unpad_input(
            input_ids, attention_mask
        )
        batch_size, seqlen = input_ids.shape[:2]
    x = PipeTransferData(cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
    ys = [PipeCacheData(packed_input_ids=packed_input_ids)] + [
        PipeCacheData() for _ in range(self.config.n_layers + 1)
    ]
    scores = ReaLModel.forward(self, x, ys)[0].pp_output
    if build_packed:
        scores = pad_input(scores, indices, batch_size, seqlen)
    return DuckModelOutput(logits=scores)


def add_helper_functions(m: ReaLModel):
    m.forward = functools.partial(forward_helper, m)
    m.generate = functools.partial(generate_helper, m)
    return m


def make_real_model(
    name: ModelName,
    device: torch.device,
    model_path: str,
    is_critic: bool,
    init_from_scratch: bool,
    init_critic_from_actor: bool,
    dtype: Optional[str] = None,
    hf_model_family: Optional[str] = None,
) -> model_api.Model:
    if dtype == "fp16" or dtype == None:
        dtype = torch.float16
    elif dtype == "bf16":
        dtype = torch.bfloat16
    elif dtype == "fp32":
        dtype = torch.float32
    else:
        raise NotImplementedError(f"Unsupported dtype {dtype}")

    tokenizer = model_api.load_hf_tokenizer(model_path)
    mconfig = getattr(ReaLModel, f"config_from_{hf_model_family}")(
        model_path=model_path,
        is_critic=is_critic or init_critic_from_actor,
    )
    m = ReaLModel(mconfig, dtype=dtype, device=device, hf_model_family=hf_model_family)

    if not init_from_scratch:
        m._instantiation_hooks.append(
            lambda: getattr(m, f"from_{hf_model_family}")(
                load_dir=model_path, init_critic_from_actor=init_critic_from_actor
            )
        )

    if constants.pipe_parallel_world_size() == 1:
        m = add_helper_functions(m)
    return model_api.Model(name, m, tokenizer, device, dtype=dtype)


model_api.register_model("real_model", make_real_model)
