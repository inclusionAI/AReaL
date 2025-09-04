import os
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_F
from tensordict import TensorDict
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)
from torch.distributed.device_mesh import init_device_mesh
from transformers import AutoProcessor, PreTrainedTokenizerFast

from areal.api.cli_args import TrainEngineConfig
from areal.api.engine_api import FinetuneSpec
from areal.api.io_struct import ParamSpec, SaveLoadMeta, WeightUpdateMeta
from areal.engine.base_hf_engine import BaseHFEngine
from areal.models.sharding.fsdp_ulysses import FSDPUlyssesShardingManager
from areal.models.transformers.ulyssess_patch import apply_monkey_patch
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
    apply_fsdp2,
    create_fsdp_device_mesh,
    fsdp2_clip_grad_norm_,
    fsdp2_load_full_state_dict,
)
from areal.utils.save_load import get_state_dict_from_repo_id_or_path
from areal.utils.ulysses import (
    get_ulysses_sequence_parallel_group,
    get_ulysses_sequence_parallel_rank,
    get_ulysses_sequence_parallel_world_size,
    ulysses_pad,
    ulysses_pad_and_slice_inputs,
)

logger = logging.getLogger("FSDPEngine")


class FSDPEngine(BaseHFEngine):
    def __init__(self, config: TrainEngineConfig):
        super().__init__(config)
        # FSDP options
        self.mixed_precision_policy = None
        self.device_mesh = None
        self.cpu_offload = None

    def initialize(self, addr: str | None, ft_spec: FinetuneSpec | None):
        # Initialize distributed enviroments and load model.
        assert addr is None, "FSDPEngine does not support remote initialization."
        assert ft_spec is not None, "FSDPEngine requires FinetuneSpec to initialize."
        assert pkg_version.is_version_greater_or_equal(
            "torch", "2.4.0"
        ), f"areal only supports FSDP2, which requires torch>=2.4.0"

        # Create process group
        self.create_process_group()

        # Create device mesh
        self.device_mesh = create_fsdp_device_mesh(self.world_size, self.world_size)

        # Create Ulysses device mesh
        self.ulysses_device_mesh = None
        self.ulysses_sp_size = self.config.fsdp.ulysses_sp_size
        if self.world_size % self.ulysses_sp_size != 0:
            raise ValueError(
                f"FSDP's world_size ({self.world_size}) must be divisible by ulysses_sp_size ({self.ulysses_sp_size})!"
            )
        dp = self.world_size // self.ulysses_sp_size
        if self.ulysses_sp_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                "cuda",
                mesh_shape=(dp, self.ulysses_sp_size),
                mesh_dim_names=("dp", "sp"),
            )
            logger.info(
                f"Created FSDP Engine with Ulysses sequence parallelism (SP={self.ulysses_sp_size})"
            )

        # This sharding manager is mainly used to save/restore distributed groups
        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(
            self.ulysses_device_mesh
        )

        # Create device model
        self.create_device_model()

        # Monkey patch: replace attention's forward() with Ulysses variant.
        apply_monkey_patch(
            model=self.model,
            ulysses_sp_size=self.ulysses_sp_size,
        )

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
            "mesh": self.device_mesh,
            "mp_policy": self.mixed_precision_policy,
            "offload_policy": self.cpu_offload,
            "reshard_after_forward": True,
        }
        tik = time.perf_counter()
        apply_fsdp2(self.model, fsdp_kwargs, self.config.fsdp.wrap_policy)
        logger.info(f"Applying FSDP2 time: {time.perf_counter() - tik}")

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
        if meta.type == "nccl":
            if not self.weight_update_group_initialized:
                self._init_distributed_weight_update(meta)
            self._update_weights_from_distributed(meta.nccl_param_specs)
            dist.barrier(device_ids=[self.device.index])
            torch.cuda.synchronize()
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
            self.weight_update_group = init_custom_process_group(
                backend="nccl",
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
                    logger.debug(f"Broadcasting {name} with shape {tensor.shape}")
                    dist.broadcast(tensor, src=0, group=self.weight_update_group)
                del tensor
            dist.barrier(device_ids=[self.device.index])
            torch.cuda.synchronize()

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

        self.optimizer.zero_grad()
        mb_list = self.prepare_mb_list(input_)

        total_loss_weight = torch.tensor(
            sum([loss_weight_fn(mb) for mb in mb_list.mbs]),
            dtype=torch.float32,
            device=self.device,
        )
        assert total_loss_weight != 0
        dist.all_reduce(total_loss_weight)

        # Process microbatches with gradient accumulation
        if self.ulysses_sp_size > 1:
            with self.ulysses_sharding_manager:
                get_ulysses_sequence_parallel_rank()
                sp_world_size = get_ulysses_sequence_parallel_world_size()
                sp_group = get_ulysses_sequence_parallel_group()
                dp_world_size = self.world_size // sp_world_size

                mb_lists = [None for _ in range(sp_world_size)]
                dist.all_gather_object(mb_lists, mb_list, group=sp_group)
                mb_lists = [mb.to(self.device) for mb in mb_lists]

                for src in range(sp_world_size):
                    for i, (pad_length, padded_mb_input, mb_input) in enumerate(
                        zip(
                            mb_lists[src].padding_lengths,
                            mb_lists[src].padded_mbs,
                            mb_lists[src].mbs,
                        )
                    ):
                        self.model.set_requires_gradient_sync(
                            src == sp_world_size - 1 and i == len(mb_lists[src].mbs) - 1
                        )

                        input_ids = padded_mb_input["input_ids"]
                        position_ids = padded_mb_input.get("position_ids", None)

                        if self.is_vision_model:
                            # NOTE: For vision models, inputs_embeds will be sliced instead of input_ids.
                            #       Please refer to patch_vlm_for_ulysses_input_slicing() in
                            #       areal/models/transformers/ulyssess_patch.py
                            (
                                ulysses_input_ids,
                                ulysses_position_ids,
                                ulysses_pad_size,
                            ) = ulysses_pad(
                                input_ids, position_ids, sp_size=sp_world_size
                            )
                        else:
                            # Pad and slice the inputs
                            (
                                ulysses_input_ids,
                                ulysses_position_ids,
                                ulysses_pad_size,
                            ) = ulysses_pad_and_slice_inputs(
                                input_ids, position_ids, sp_size=sp_world_size
                            )

                        if not ulysses_position_ids.is_contiguous():
                            ulysses_position_ids = ulysses_position_ids.contiguous()

                        ulysses_mb_input = padded_mb_input.copy()
                        ulysses_mb_input["input_ids"] = ulysses_input_ids
                        if ulysses_position_ids is not None:
                            ulysses_mb_input["position_ids"] = ulysses_position_ids

                        # NOTE: Only null attention mask is supported
                        attention_mask = ulysses_mb_input.get("attention_mask", None)
                        assert attention_mask is not None, "Attention mask is missing"
                        assert attention_mask.get("full_attention", None) is None

                        outputs = self.model(**ulysses_mb_input)

                        logits = outputs.logits.squeeze(0)
                        gathered_logits = dist_F.all_gather(logits, group=sp_group)
                        full_logits = torch.cat(gathered_logits, dim=0)
                        if ulysses_pad_size > 0:
                            full_logits = full_logits[:-ulysses_pad_size]
                        # Remove original padding
                        if pad_length > 0:
                            full_logits = full_logits[:-pad_length]

                        loss = loss_fn(full_logits, mb_input)
                        loss_scale = loss_weight_fn(mb_input) / total_loss_weight

                        # To reverse the gradient averaging for SP groups
                        loss_scale *= dp_world_size

                        loss *= loss_scale
                        loss.backward()
        else:
            for i, (pad_length, padded_mb_input, mb_input) in enumerate(
                zip(mb_list.padding_lengths, mb_list.padded_mbs, mb_list.mbs)
            ):
                self.model.set_requires_gradient_sync(i == len(mb_list.mbs) - 1)
                outputs = self.model(**padded_mb_input)

                logits = outputs.logits.squeeze(0)
                logits = logits[:-pad_length] if pad_length > 0 else logits
                loss = loss_fn(logits, mb_input)
                loss_scale = loss_weight_fn(mb_input) / total_loss_weight

                # Scale loss for accumulation
                # Revert gradient averaging across dp ranks
                loss_scale *= self.world_size

                loss *= loss_scale
                loss.backward()

        # NOTE: grad norm clip function is different

        grad_norm = fsdp2_clip_grad_norm_(
            self.model.parameters(), max_norm=self.optimizer_config.gradient_clipping
        )

        if not torch.isfinite(grad_norm):
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
        input_ = input_.to(self.device)
        mb_list = self.prepare_mb_list(input_)
        total_loss_weight = torch.tensor(
            sum([loss_weight_fn(mb) for mb in mb_list.mbs]), dtype=torch.float32
        )
        assert total_loss_weight != 0

        total_loss = 0.0
        total_weight = 0.0

        if self.ulysses_sp_size > 1:
            with self.ulysses_sharding_manager:
                get_ulysses_sequence_parallel_rank()
                sp_world_size = get_ulysses_sequence_parallel_world_size()
                sp_group = get_ulysses_sequence_parallel_group()

                mb_lists = [None for _ in range(sp_world_size)]
                dist.all_gather_object(mb_lists, mb_list, group=sp_group)
                mb_lists = [mb.to(self.device) for mb in mb_lists]

                for src in range(sp_world_size):
                    for pad_length, padded_mb_input, mb_input in zip(
                        mb_lists[src].padding_lengths,
                        mb_lists[src].padded_mbs,
                        mb_lists[src].mbs,
                    ):
                        input_ids = padded_mb_input["input_ids"]
                        position_ids = padded_mb_input.get("position_ids", None)

                        if self.is_vision_model:
                            # NOTE: For vision models, inputs_embeds will be sliced instead of input_ids.
                            #       Please refer to patch_vlm_for_ulysses_input_slicing() in
                            #       areal/models/transformers/ulyssess_patch.py
                            (
                                ulysses_input_ids,
                                ulysses_position_ids,
                                ulysses_pad_size,
                            ) = ulysses_pad(
                                input_ids, position_ids, sp_size=sp_world_size
                            )
                        else:
                            # Pad and slice the inputs
                            (
                                ulysses_input_ids,
                                ulysses_position_ids,
                                ulysses_pad_size,
                            ) = ulysses_pad_and_slice_inputs(
                                input_ids, position_ids, sp_size=sp_world_size
                            )

                        if not ulysses_position_ids.is_contiguous():
                            ulysses_position_ids = ulysses_position_ids.contiguous()

                        ulysses_mb_input = padded_mb_input.copy()
                        ulysses_mb_input["input_ids"] = ulysses_input_ids
                        if ulysses_position_ids is not None:
                            ulysses_mb_input["position_ids"] = ulysses_position_ids

                        outputs = self.model(**ulysses_mb_input)

                        logits = outputs.logits.squeeze(0)
                        gathered_logits = dist_F.all_gather(logits, group=sp_group)
                        full_logits = torch.cat(gathered_logits, dim=0)
                        if ulysses_pad_size > 0:
                            full_logits = full_logits[:-ulysses_pad_size]
                        # Remove original padding
                        if pad_length > 0:
                            full_logits = full_logits[:-pad_length]

                        loss = loss_fn(full_logits, mb_input)
                        loss_scale = loss_weight_fn(mb_input) / total_loss_weight
                        total_loss += loss.item() * loss_scale
                        total_weight += loss_scale
        else:
            for pad_length, padded_mb_input, mb_input in zip(
                mb_list.padding_lengths, mb_list.padded_mbs, mb_list.mbs
            ):
                outputs = self.model(**padded_mb_input)
                logits = outputs.logits.squeeze(0)
                logits = logits[:-pad_length] if pad_length > 0 else logits
                loss = loss_fn(logits, mb_input)

                # Simple weight calculation (could be improved)
                loss_scale = loss_weight_fn(mb_input) / total_loss_weight
                total_loss += loss.item() * loss_scale
                total_weight += loss_scale

        return torch.tensor(total_loss / total_weight)

    @torch.no_grad()
    def forward(
        self,
        input_: TensorDict,
        output_seqlens: List[int] | None = None,
        post_hook: Callable[[torch.Tensor, TensorDict], Any] | None = None,
        aggregate_fn: Callable[[List[Any]], Any] = torch.cat,
    ) -> Any | None:
        """Forward pass with optional post-processing."""
        input_ = input_.to(self.device)
        cu_seqlens = pack_tensor_dict(input_)["cu_seqlens"]
        mb_list = self.prepare_mb_list(input_)

        if output_seqlens is None:
            output_seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).cpu().numpy().tolist()

        results = []

        if self.ulysses_sp_size > 1:
            with self.ulysses_sharding_manager:
                sp_rank = get_ulysses_sequence_parallel_rank()
                sp_world_size = get_ulysses_sequence_parallel_world_size()
                sp_group = get_ulysses_sequence_parallel_group()

                mb_lists = [None for _ in range(sp_world_size)]
                dist.all_gather_object(mb_lists, mb_list, group=sp_group)
                mb_lists = [mb.to(self.device) for mb in mb_lists]

                for src in range(sp_world_size):
                    for pad_length, padded_mb_input, mb_input in zip(
                        mb_lists[src].padding_lengths,
                        mb_lists[src].padded_mbs,
                        mb_lists[src].mbs,
                    ):
                        input_ids = padded_mb_input["input_ids"]
                        position_ids = padded_mb_input.get("position_ids", None)

                        if self.is_vision_model:
                            # NOTE: For vision models, inputs_embeds will be sliced instead of input_ids.
                            #       Please refer to patch_vlm_for_ulysses_input_slicing() in
                            #       areal/models/transformers/ulyssess_patch.py
                            (
                                ulysses_input_ids,
                                ulysses_position_ids,
                                ulysses_pad_size,
                            ) = ulysses_pad(
                                input_ids, position_ids, sp_size=sp_world_size
                            )
                        else:
                            # Pad and slice the inputs
                            (
                                ulysses_input_ids,
                                ulysses_position_ids,
                                ulysses_pad_size,
                            ) = ulysses_pad_and_slice_inputs(
                                input_ids, position_ids, sp_size=sp_world_size
                            )

                        if not ulysses_position_ids.is_contiguous():
                            ulysses_position_ids = ulysses_position_ids.contiguous()

                        ulysses_mb_input = padded_mb_input.copy()
                        ulysses_mb_input["input_ids"] = ulysses_input_ids
                        if ulysses_position_ids is not None:
                            ulysses_mb_input["position_ids"] = ulysses_position_ids

                        outputs = self.model(**ulysses_mb_input)

                        logits = outputs.logits.squeeze(0)
                        gathered_logits = dist_F.all_gather(logits, group=sp_group)
                        full_logits = torch.cat(gathered_logits, dim=0)
                        if ulysses_pad_size > 0:
                            full_logits = full_logits[:-ulysses_pad_size]
                        # Remove original padding
                        if pad_length > 0:
                            full_logits = full_logits[:-pad_length]

                        if post_hook:
                            result = post_hook(full_logits, mb_input)
                        else:
                            result = full_logits

                        if sp_rank == src:
                            results.append(result)
        else:
            for pad_length, padded_mb_input, mb_input in zip(
                mb_list.padding_lengths, mb_list.padded_mbs, mb_list.mbs
            ):
                outputs = self.model(**padded_mb_input)
                logits = outputs.logits.squeeze(0)
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
