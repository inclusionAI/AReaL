import dataclasses
import math
import os
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_F
import torch.nn as nn
from tensordict import TensorDict
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    ParallelStyle,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)
from transformers import AutoProcessor, PreTrainedTokenizerFast

from areal.api.alloc_mode import FSDPParallelStrategy, ParallelStrategy
from areal.api.cli_args import TrainEngineConfig
from areal.api.io_struct import FinetuneSpec, ParamSpec, SaveLoadMeta, WeightUpdateMeta
from areal.engine.base_hf_engine import BaseHFEngine
from areal.models.transformers.ulyssess_patch import apply_monkey_patch
from areal.platforms import current_platform
from areal.utils import datapack, logging, name_resolve, names, pkg_version
from areal.utils.data import (
    pack_tensor_dict,
    pad_and_stack_tensors_along_first_dim,
    reorder_list,
    unpack_sequence,
)
from areal.utils.distributed import init_custom_process_group
from areal.utils.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    NoParallel,
    apply_fsdp2,
    fsdp2_clip_grad_norm,
    fsdp2_load_full_state_dict,
)
from areal.utils.model import VALID_VISION_MODELS
from areal.utils.save_load import get_state_dict_from_repo_id_or_path
from areal.utils.ulysses import (
    set_ulysses_sequence_parallel_group,
    ulysses_pad,
    ulysses_pad_and_slice_inputs,
)
from areal.platforms import is_npu_available


class FSDPEngine(BaseHFEngine):
    def __init__(self, config: TrainEngineConfig):
        super().__init__(config)
        # FSDP options
        self.fsdp_tp_device_mesh = None
        self.mixed_precision_policy = None
        self.cpu_offload = None

        self.dp_world_size = None
        self.sp_world_size = None
        self.tp_world_size = None

        self.dp_group = None
        self.sp_group = None
        self.mp_group = None

        self.rank = None
        self.dp_head = None
        self.dp_rank = None

    @property
    def data_parallel_group(self) -> dist.ProcessGroup:
        assert self.dp_group is not None
        return self.dp_group

    @property
    def data_parallel_rank(self) -> int:
        assert self.dp_rank is not None
        return self.dp_rank

    @property
    def data_parallel_world_size(self) -> int:
        assert self.dp_world_size is not None
        return self.dp_world_size

    def current_data_parallel_head(self) -> int:
        assert self.dp_head is not None
        return self.dp_head

    def is_data_parallel_head(self) -> bool:
        return self.rank == self.dp_head

    @property
    def context_and_model_parallel_group(self) -> dist.ProcessGroup:
        assert self.mp_group is not None
        return self.mp_group

    def _make_parallel_strategy(
        self, parallel_strategy: ParallelStrategy
    ) -> FSDPParallelStrategy:
        return FSDPParallelStrategy(
            **dataclasses.asdict(parallel_strategy),
        )

    def create_process_group(self, parallel_strategy: ParallelStrategy | None = None):
        super().create_process_group(parallel_strategy)
        if parallel_strategy is None:
            parallel_strategy = ParallelStrategy()

        self.logger = logging.getLogger(f"[FSDP Engine Rank {dist.get_rank()}]")

        self.parallel_strategy = self._make_parallel_strategy(parallel_strategy)

        self.dp_world_size = self.parallel_strategy.data_parallel_size
        # This is Ulysses sequence parallelism
        self.sp_world_size = self.parallel_strategy.context_parallel_size
        self.tp_world_size = self.parallel_strategy.tensor_parallel_size

        self.logger.info(
            f"Initializing device mesh with mode d{self.dp_world_size}s{self.sp_world_size}t{self.tp_world_size}."
        )

        self.fsdp_tp_device_mesh = init_device_mesh(
            current_platform.device_type,
            mesh_shape=(self.dp_world_size * self.sp_world_size, self.tp_world_size),
            mesh_dim_names=("fsdp", "tp"),
        )

        nd_device_mesh = init_device_mesh(
            current_platform.device_type,
            mesh_shape=(self.dp_world_size, self.sp_world_size, self.tp_world_size),
            mesh_dim_names=("dp", "sp", "tp"),
        )

        self.dp_group = nd_device_mesh["dp"].get_group()
        self.sp_group = nd_device_mesh["sp"].get_group()

        nd_device_mesh["sp", "tp"]._flatten(mesh_dim_name="mp")

        self.mp_group = nd_device_mesh["mp"].get_group()

        self.rank = dist.get_rank()

        self.dp_head = nd_device_mesh["mp"].mesh[0].item()
        self.dp_rank = dist.get_rank(self.dp_group)

        self.logger.info(f"Data parallel head {self.dp_head} and rank {self.dp_rank}")

    def apply_tensor_parallel(self, device_mesh: DeviceMesh):
        try:
            num_attention_heads, num_key_value_heads = (
                self.model.config.num_attention_heads,
                self.model.config.num_key_value_heads,
            )
        except AttributeError:
            num_attention_heads, num_key_value_heads = (
                self.model.config.text_config.num_attention_heads,
                self.model.config.text_config.num_key_value_heads,
            )

        if (
            num_attention_heads % self.tp_world_size != 0
            or num_key_value_heads % self.tp_world_size != 0
        ):
            raise ValueError(
                f"num_attention_heads {num_attention_heads} and num_key_value_heads {num_key_value_heads} must be divisible by tensor_parallel_size {self.tp_world_size}"
            )

        # For root module
        root_tp_plan: dict[str, ParallelStyle] = {
            # All-gather
            "lm_head": ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Replicate(),
            ),
        }

        # For model or model.language_model
        model_tp_plan: dict[str, ParallelStyle] = {
            "embed_tokens": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
                use_local_output=False,
            ),
            "layers.*.input_layernorm": SequenceParallel(),
            # All-gather
            "layers.*.self_attn": PrepareModuleInput(
                input_kwarg_layouts={"hidden_states": Shard(1)},
                desired_input_kwarg_layouts={"hidden_states": Replicate()},
            ),
            "layers.*.self_attn.q_proj": ColwiseParallel(),
            "layers.*.self_attn.k_proj": ColwiseParallel(),
            "layers.*.self_attn.v_proj": ColwiseParallel(),
            # special q/k norm for qwen3
            "layers.*.self_attn.q_norm": NoParallel(),
            "layers.*.self_attn.k_norm": NoParallel(),
            # Reduce in RowwiseParallel, Scatter by Shard(1)
            "layers.*.self_attn.o_proj": RowwiseParallel(
                output_layouts=Shard(1),
                use_local_output=False,
            ),
            "layers.*.post_attention_layernorm": SequenceParallel(),
            # All-gather
            "layers.*.mlp": PrepareModuleInput(
                input_layouts=Shard(1),
                desired_input_layouts=Replicate(),
            ),
            "layers.*.mlp.gate_proj": ColwiseParallel(),
            "layers.*.mlp.up_proj": ColwiseParallel(),
            # Reduce in RowwiseParallel, Scatter by Shard(1)
            "layers.*.mlp.down_proj": RowwiseParallel(
                output_layouts=Shard(1),
                use_local_output=False,
            ),
            "norm": SequenceParallel(),
        }

        if not isinstance(self.model.model, nn.Module):
            raise RuntimeError("Model does not have the required submodule 'model'.")

        if self.is_vision_model:
            if self.model_config.model_type not in VALID_VISION_MODELS:
                self.logger.warning(
                    f"Vision model type {self.model_config.model_type} not in supported list {VALID_VISION_MODELS}."
                )

            if isinstance(self.model.model.language_model, nn.Module):
                # For vision-language models, avoid sharding the embedding layer because
                # the visual components access it without tensor parallelism support.
                # Instead, configure the first transformer layer to handle input
                # sharding properly.
                model_tp_plan.pop("embed_tokens", None)
                model_tp_plan["layers.0"] = PrepareModuleInput(
                    input_layouts=Replicate(),
                    desired_input_layouts=Shard(1),
                )

                parallelize_module(
                    self.model.model.language_model,
                    device_mesh=device_mesh,
                    parallelize_plan=model_tp_plan,
                )
            else:
                self.logger.warning(
                    f"Vision model does not have the required submodule 'model.language_model'."
                )
        else:
            parallelize_module(
                self.model.model,
                device_mesh=device_mesh,
                parallelize_plan=model_tp_plan,
            )

        self.model = parallelize_module(
            self.model,
            device_mesh=device_mesh,
            parallelize_plan=root_tp_plan,
        )

    def initialize(self, addr: str | None, ft_spec: FinetuneSpec | None):
        # Initialize distributed enviroments and load model.
        assert addr is None, "FSDPEngine does not support remote initialization."
        assert ft_spec is not None, "FSDPEngine requires FinetuneSpec to initialize."
        assert pkg_version.is_version_greater_or_equal(
            "torch", "2.4.0"
        ), f"areal only supports FSDP2, which requires torch>=2.4.0"

        assert self.fsdp_tp_device_mesh is not None
        assert self.sp_world_size is not None

        # Create device model
        self.create_device_model()

        # Monkey patch: replace attention's forward() with Ulysses variant.
        apply_monkey_patch(
            model=self.model,
            ulysses_sp_size=self.sp_world_size,
        )

        if self.tp_world_size > 1:
            self.apply_tensor_parallel(self.fsdp_tp_device_mesh["tp"])

        # sharding_strategy = ShardingStrategy.FULL_SHARD
        # Simple auto wrap policy
        self.mixed_precision_policy = MixedPrecisionPolicy(
            param_dtype=getattr(torch, self.config.dtype),
            reduce_dtype=getattr(torch, self.config.grad_reduce_dtype),
            cast_forward_inputs=True,
        )
        self.cpu_offload = (
            CPUOffloadPolicy() if self.config.fsdp.offload_params else None
        )
        fsdp_kwargs = {
            "mesh": self.fsdp_tp_device_mesh["fsdp"],
            "mp_policy": self.mixed_precision_policy,
            "offload_policy": self.cpu_offload,
            "reshard_after_forward": True,
        }
        tik = time.perf_counter()
        apply_fsdp2(self.model, fsdp_kwargs, self.config.fsdp.wrap_policy)
        self.logger.info(f"Applying FSDP2 time: {time.perf_counter() - tik}")

        self.create_optimizer(ft_spec)
        self.initialized = True

    def save(self, meta: SaveLoadMeta):
        if meta.weight_format == "hf":
            self._save_model_to_hf(meta.path, meta.tokenizer, meta.processor)
        elif meta.weight_format == "dcp":
            # TODO: implement DCP save/load for FSDP
            raise NotImplementedError("DCP format saving is not implemented yet. ")
        else:
            raise ValueError(f"Unknown weight format {meta.weight_format}. ")

        if meta.with_optim:
            self.save_optimizer_state(meta.path)

    def load(self, meta: SaveLoadMeta):
        if meta.weight_format == "hf":
            self._load_model_from_hf(meta.path)
        elif meta.weight_format == "dcp":
            # TODO: implement DCP save/load for FSDP
            raise NotImplementedError("DCP format loading is not implemented yet. ")
        else:
            raise ValueError(f"Unknown weight format {meta.weight_format}. ")

        if meta.with_optim:
            self.load_optimizer_state(meta.path)

    def _save_model_to_hf(
        self,
        path: str,
        tokenizer: Optional[PreTrainedTokenizerFast],
        processor: Optional[AutoProcessor],
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

        dist.barrier(device_ids=[self.device.index])

    def _load_model_from_hf(self, path: str):
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

    def upload_weights(self, meta: WeightUpdateMeta):
        if meta.type == "nccl" or meta.type == "hccl":
            if not self.weight_update_group_initialized:
                self._init_distributed_weight_update(meta)
            self._update_weights_from_distributed(meta.nccl_param_specs)
            dist.barrier(device_ids=[self.device.index])
            current_platform.synchronize()
        elif meta.type == "disk":
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
        else:
            raise ValueError(f"Unknown weight update type {meta.type}")

    def _init_distributed_weight_update(self, meta: WeightUpdateMeta):
        # NOTE: Processes launched with torchrun will set the following env var to True,
        # which blocks creating another TCP store for weight update.
        os.environ["TORCHELASTIC_USE_AGENT_STORE"] = str(False)
        if dist.get_rank() == 0:
            backend = current_platform.communication_backend
            self.weight_update_group = init_custom_process_group(
                backend=backend,
                world_size=meta.alloc_mode.gen.world_size + 1,
                init_method=f"tcp://{meta.nccl_master_address}:{meta.nccl_master_port}",
                rank=0,
                group_name=meta.nccl_group_name,
            )
            # NOTE: sglang v0.4.9.post2 or later does not have the barrier call
        self.weight_update_group_initialized = True

    def _update_weights_from_distributed(
        self, grouped_param_specs: List[List[ParamSpec]]
    ):
        """Broadcast parameters (chunked) from rank 0 (FSDP2 compatible)."""

        named_parameters = dict(self.get_model_name_parameters())
        for param_specs in grouped_param_specs:
            for param_spec in param_specs:
                name = param_spec.name
                param = named_parameters[name]
                if isinstance(param.data, DTensor):
                    tensor = param.data.full_tensor()
                else:
                    tensor = param.data
                if dist.get_rank() == 0:
                    self.logger.debug(f"Broadcasting {name} with shape {tensor.shape}")
                    dist.broadcast(tensor, src=0, group=self.weight_update_group, async_op=False)
                del tensor
            dist.barrier(device_ids=[self.device.index])
            current_platform.synchronize()

    def _bin_pack_param_specs(
        self, param_specs: List[ParamSpec], chunked_mem_mb=1024
    ) -> List[List[ParamSpec]]:
        sizes = [param_spec.size for param_spec in param_specs]
        chunked_mem_bytes = max(chunked_mem_mb * 1024 * 1024, max(sizes) + 10)
        group_indices = datapack.ffd_allocate(sizes, chunked_mem_bytes, 1)
        grouped_param_specs = [
            [param_specs[i] for i in group_index] for group_index in group_indices
        ]
        return grouped_param_specs

    def get_param_specs(
        self, weight_chunked_mem_mb: int = 1024
    ) -> List[List[ParamSpec]]:
        param_specs = []
        for name, param in self.get_model_name_parameters():
            if isinstance(param.data, DTensor):
                tensor = param.data.full_tensor()
            else:
                tensor = param.data
            param_specs.append(
                ParamSpec(
                    name=name,
                    shape=tuple(tensor.shape),
                    dtype=str(tensor.dtype).split("torch.")[1],
                )
            )
            del tensor  # free memory if full_tensor was created
        return self._bin_pack_param_specs(
            param_specs, chunked_mem_mb=weight_chunked_mem_mb
        )

    def train_batch(
        self,
        input_: TensorDict,
        loss_fn: Callable[[torch.Tensor, TensorDict], torch.Tensor],
        loss_weight_fn: Callable[[TensorDict], float],
    ) -> Dict[str, float]:
        """Train on a batch using gradient accumulation."""
        input_ = input_.to(self.device)
        assert self.optimizer is not None
        assert self.optimizer_config is not None
        assert self.lr_scheduler is not None
        assert self.fsdp_tp_device_mesh is not None
        set_ulysses_sequence_parallel_group(self.sp_group)

        self.optimizer.zero_grad()
        mb_list = self.prepare_mb_list(input_)
        mb_list = mb_list.to(self.device)

        total_loss_weight = (
            sum([loss_weight_fn(mb) for mb in mb_list.mbs])
            .detach()
            .clone()
            .to(dtype=torch.float32, device=self.device)
        )
        assert total_loss_weight != 0
        dist.all_reduce(total_loss_weight, group=self.dp_group)

        # Process microbatches with gradient accumulation
        for i, (pad_length, padded_mb_input, mb_input) in enumerate(
            zip(mb_list.padding_lengths, mb_list.padded_mbs, mb_list.mbs)
        ):
            if self.sp_world_size > 1:
                input_ids = padded_mb_input["input_ids"]
                position_ids = padded_mb_input.get("position_ids", None)

                if self.is_vision_model:
                    (
                        ulysses_input_ids,
                        ulysses_position_ids,
                        ulysses_pad_size,
                    ) = ulysses_pad(input_ids, position_ids, sp_size=self.sp_world_size)
                else:
                    # Pad and slice the inputs
                    (
                        ulysses_input_ids,
                        ulysses_position_ids,
                        ulysses_pad_size,
                    ) = ulysses_pad_and_slice_inputs(
                        input_ids,
                        position_ids,
                        sp_size=self.sp_world_size,
                    )

                if (
                    ulysses_position_ids is not None
                    and not ulysses_position_ids.is_contiguous()
                ):
                    ulysses_position_ids = ulysses_position_ids.contiguous()

                inputs = padded_mb_input.copy()
                inputs["input_ids"] = ulysses_input_ids
                if ulysses_position_ids is not None:
                    inputs["position_ids"] = ulysses_position_ids
            else:
                inputs = padded_mb_input

            outputs = self.model(**inputs)

            logits = outputs.logits.squeeze(0)
            if self.sp_world_size > 1:
                # Gather and remove Ulysses padding
                gathered_logits = dist_F.all_gather(logits, group=self.sp_group)
                logits = torch.cat(gathered_logits, dim=0)
                logits = logits[:-ulysses_pad_size] if ulysses_pad_size > 0 else logits
            # Remove original padding
            logits = logits[:-pad_length] if pad_length > 0 else logits
            loss = loss_fn(logits, mb_input)
            loss_scale = loss_weight_fn(mb_input) / total_loss_weight

            # Scale loss for accumulation
            # To reverse the gradient averaging for SP groups
            loss_scale *= self.dp_world_size

            loss *= loss_scale
            loss.backward()

        # NOTE: grad norm clip function is different
        grad_norm = fsdp2_clip_grad_norm(
            list(self.model.parameters()),
            self.fsdp_tp_device_mesh,
            max_norm=self.optimizer_config.gradient_clipping,
        )

        if not math.isfinite(grad_norm):
            self.optimizer.zero_grad()
            update_successful = False
        else:
            self.optimizer.step()
            update_successful = True

        current_lr = self.lr_scheduler.get_last_lr()[0]
        return dict(
            update_successful=float(update_successful),
            grad_norm=float(grad_norm) if grad_norm is not None else float("nan"),
            lr=current_lr,
        )

    @torch.no_grad()
    def eval_batch(
        self,
        input_: TensorDict,
        loss_fn: Callable[[torch.Tensor, TensorDict], torch.Tensor],
        loss_weight_fn: Callable[[TensorDict], float],
    ) -> torch.Tensor | None:
        """Evaluate on a batch."""
        mb_list = self.prepare_mb_list(input_)
        mb_list = mb_list.to(self.device)
        set_ulysses_sequence_parallel_group(self.sp_group)

        total_loss_weight = (
            sum([loss_weight_fn(mb) for mb in mb_list.mbs])
            .detach()
            .clone()
            .to(dtype=torch.float32)
        )
        assert total_loss_weight != 0
        dist.all_reduce(total_loss_weight, group=self.dp_group)

        total_loss = torch.zeros(1, device=self.device, dtype=torch.float32)

        for pad_length, padded_mb_input, mb_input in zip(
            mb_list.padding_lengths, mb_list.padded_mbs, mb_list.mbs
        ):
            if self.sp_world_size > 1:
                input_ids = padded_mb_input["input_ids"]
                position_ids = padded_mb_input.get("position_ids", None)

                if self.is_vision_model:
                    (
                        ulysses_input_ids,
                        ulysses_position_ids,
                        ulysses_pad_size,
                    ) = ulysses_pad(input_ids, position_ids, sp_size=self.sp_world_size)
                else:
                    # Pad and slice the inputs
                    (
                        ulysses_input_ids,
                        ulysses_position_ids,
                        ulysses_pad_size,
                    ) = ulysses_pad_and_slice_inputs(
                        input_ids,
                        position_ids,
                        sp_size=self.sp_world_size,
                    )

                if (
                    ulysses_position_ids is not None
                    and not ulysses_position_ids.is_contiguous()
                ):
                    ulysses_position_ids = ulysses_position_ids.contiguous()

                inputs = padded_mb_input.copy()
                inputs["input_ids"] = ulysses_input_ids
                if ulysses_position_ids is not None:
                    inputs["position_ids"] = ulysses_position_ids
            else:
                inputs = padded_mb_input

            outputs = self.model(**inputs)

            logits = outputs.logits.squeeze(0)
            if self.sp_world_size > 1:
                # Gather and remove Ulysses padding
                gathered_logits = dist_F.all_gather(logits, group=self.sp_group)
                logits = torch.cat(gathered_logits, dim=0)
                logits = logits[:-ulysses_pad_size] if ulysses_pad_size > 0 else logits
            # Remove original padding
            logits = logits[:-pad_length] if pad_length > 0 else logits
            loss = loss_fn(logits, mb_input)

            # Simple weight calculation (could be improved)
            loss_scale = loss_weight_fn(mb_input) / total_loss_weight
            # eval_batch does not run backward, the grad will not be averaged over DP group
            # so we shouldn't multiple dp_size in loss_scale
            total_loss += loss.clone().detach() * loss_scale

        dist.all_reduce(total_loss, group=self.dp_group)

        return total_loss

    @torch.no_grad()
    def forward(
        self,
        input_: TensorDict,
        output_seqlens: List[int] | None = None,
        post_hook: Callable[[torch.Tensor, TensorDict], Any] | None = None,
        aggregate_fn: Callable[[List[Any]], Any] = torch.cat,
    ) -> Any | None:
        """Forward pass with optional post-processing."""
        cu_seqlens = pack_tensor_dict(input_)["cu_seqlens"]
        mb_list = self.prepare_mb_list(input_)
        mb_list = mb_list.to(self.device)
        set_ulysses_sequence_parallel_group(self.sp_group)

        if output_seqlens is None:
            output_seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).cpu().numpy().tolist()

        results = []

        for pad_length, padded_mb_input, mb_input in zip(
            mb_list.padding_lengths, mb_list.padded_mbs, mb_list.mbs
        ):
            if self.sp_world_size > 1:
                input_ids = padded_mb_input["input_ids"]
                position_ids = padded_mb_input.get("position_ids", None)

                if self.is_vision_model:
                    (
                        ulysses_input_ids,
                        ulysses_position_ids,
                        ulysses_pad_size,
                    ) = ulysses_pad(input_ids, position_ids, sp_size=self.sp_world_size)
                else:
                    # Pad and slice the inputs
                    (
                        ulysses_input_ids,
                        ulysses_position_ids,
                        ulysses_pad_size,
                    ) = ulysses_pad_and_slice_inputs(
                        input_ids,
                        position_ids,
                        sp_size=self.sp_world_size,
                    )

                if (
                    ulysses_position_ids is not None
                    and not ulysses_position_ids.is_contiguous()
                ):
                    ulysses_position_ids = ulysses_position_ids.contiguous()

                inputs = padded_mb_input.copy()
                inputs["input_ids"] = ulysses_input_ids
                if ulysses_position_ids is not None:
                    inputs["position_ids"] = ulysses_position_ids
            else:
                inputs = padded_mb_input

            outputs = self.model(**inputs)

            logits = outputs.logits.squeeze(0)
            if self.sp_world_size > 1:
                # Gather and remove Ulysses padding
                gathered_logits = dist_F.all_gather(logits, group=self.sp_group)
                logits = torch.cat(gathered_logits, dim=0)
                logits = logits[:-ulysses_pad_size] if ulysses_pad_size > 0 else logits
            # Remove original padding
            logits = logits[:-pad_length] if pad_length > 0 else logits

            if post_hook:
                result = post_hook(logits, mb_input)
                results.append(result)
            else:
                results.append(logits)

        res = aggregate_fn(results)
        output_seqlens = [output_seqlens[i] for i in mb_list.forward_indices]
        unpacked = unpack_sequence(res, lens=output_seqlens, dim=0)
        reordered = reorder_list(unpacked, mb_list.backward_indices)
        return pad_and_stack_tensors_along_first_dim(reordered)
