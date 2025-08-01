import os
import time
from datetime import datetime
from typing import Callable, Dict, List, Optional

import torch
import torch.distributed as dist
from tensordict import TensorDict
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)
from transformers import AutoProcessor, PreTrainedTokenizerFast

from areal.api.cli_args import TrainEngineConfig
from areal.api.engine_api import FinetuneSpec
from areal.api.io_struct import ParamSpec, SaveLoadMeta, WeightUpdateMeta
from areal.engine.base_hf_engine import BaseHFEngine
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
from realhf.base import logging, name_resolve, names, pkg_version

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
        assert pkg_version.is_version_greater_or_equal(
            "torch", "2.4.0"
        ), f"areal only supports FSDP2, which requires torch>=2.4.0"

        self.create_process_group()
        self.create_device_model()

        # Wrap with FSDP2
        # Simple auto wrap policy
        self.mixed_precision_policy = MixedPrecisionPolicy(
            param_dtype=getattr(torch, self.config.dtype),
            reduce_dtype=getattr(torch, self.config.grad_reduce_dtype),
            cast_forward_inputs=True,
        )
        self.device_mesh = create_fsdp_device_mesh(self.world_size, self.world_size)
        # sharding_strategy = ShardingStrategy.FULL_SHARD
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
            self._update_weights_from_distributed()
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
                world_size=meta.alloc_mode.gen_world_size + 1,
                init_method=f"tcp://{meta.nccl_master_address}:{meta.nccl_master_port}",
                rank=0,
                group_name=meta.nccl_group_name,
            )
            # NOTE: sglang v0.4.9.post2 or later does not have the barrier call
        self.weight_update_group_initialized = True

    def _update_weights_from_distributed(self):
        """Broadcast parameters from rank 0 (FSDP2 compatible)."""

        for name, param in self.model.named_parameters():
            if isinstance(param.data, DTensor):
                tensor = param.data.full_tensor()
            else:
                tensor = param.data
            if dist.get_rank() == 0:
                logger.debug(
                    f"Broadcasting {name} with shape {tensor.shape}", flush=True
                )
                dist.broadcast(tensor, src=0, group=self.weight_update_group)
            del tensor  # optional, for memory hygiene
        torch.cuda.empty_cache()

    def get_param_specs(self) -> List[ParamSpec]:
        param_specs = []
        for name, param in self.model.named_parameters():
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
        return param_specs

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
            sum([loss_weight_fn(mb) for mb in mb_list.mbs]), dtype=torch.float32
        )
        assert total_loss_weight != 0
        dist.all_reduce(total_loss_weight)

        # Process microbatches with gradient accumulation
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
