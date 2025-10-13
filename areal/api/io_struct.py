import os
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
from PIL.Image import Image as ImageObject
from transformers import PreTrainedTokenizerFast

from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import GenerationHyperparameters
from areal.platforms import current_platform
from areal.utils.network import find_free_ports, gethostip

if TYPE_CHECKING:
    from transformers import AutoProcessor


@dataclass
class ModelRequest:
    rid: str = field(default_factory=lambda: str(uuid.uuid4()))
    input_ids: List[int] = field(default_factory=list)
    gconfig: GenerationHyperparameters = field(
        default_factory=GenerationHyperparameters
    )
    metadata: Dict[str, Any] = field(default_factory=dict)
    # tokenizer is used for encode-decode in the inference engine
    tokenizer: Optional[PreTrainedTokenizerFast] = None

    # vlm
    image_data: Optional[List[ImageObject | str]] = field(default_factory=list)
    processor: Optional["AutoProcessor"] = None


@dataclass
class ModelResponse:
    # outputs
    input_tokens: List[int] = field(default_factory=list)
    output_tokens: List[int] = field(default_factory=list)
    output_logprobs: List[float] = field(default_factory=list)
    output_versions: List[int] = field(default_factory=list)
    stop_reason: Literal["length", "stop", "interrupt"] = "stop"
    # tokenizer is used for encode-decode in the inference engine
    tokenizer: Optional[PreTrainedTokenizerFast] = None

    # vlm
    input_images: List[ImageObject | str] = field(default_factory=list)
    processor: Optional["AutoProcessor"] = None

    # statistics
    latency: float = float("inf")
    ttft: float = float("inf")  # Time to first token
    itl: List[float] = field(default_factory=list)  # List of inter-token latencies

    @property
    def input_len(self) -> int:
        return len(self.input_tokens)

    @property
    def output_len(self) -> int:
        return len(self.output_tokens)


@dataclass
class FinetuneSpec:
    total_train_epochs: int
    dataset_size: int
    train_batch_size: int

    @property
    def total_train_steps(self):
        # assuming drop_last
        return self.total_train_epochs * (self.dataset_size // self.train_batch_size)

    @property
    def steps_per_epoch(self):
        return self.dataset_size // self.train_batch_size


@dataclass
class ParamSpec:
    name: str
    shape: Tuple
    dtype: str

    @property
    def size(self) -> int:
        """Param bytes"""
        return getattr(torch, self.dtype).itemsize * np.prod(self.shape)


@dataclass
class WeightUpdateMeta:
    type: Literal["disk", "nccl"]
    path: str | None = None
    alloc_mode: AllocationMode | None = None

    nccl_master_address: str = "127.0.0.1"
    nccl_master_port: int = 29500
    nccl_group_name: str = "update_weight_group"
    weight_chunked_mem_mb: int = 1024

    use_lora: bool = False

    @classmethod
    def from_disk(
        cls,
        experiment_name: str,
        trial_name: str,
        file_root: str,
        name: str = "default",
        use_lora: bool = False,
    ) -> "WeightUpdateMeta":
        from areal.utils.saver import Saver

        path = os.path.join(
            Saver.get_model_save_root(experiment_name, trial_name, file_root, name),
            "weight_update",
        )
        return cls(
            type="disk",
            path=path,
            use_lora=use_lora,
        )

    @classmethod
    def from_megatron_xccl(
        cls,
        allocation_mode: AllocationMode,
        nccl_group_name: str = "update_weight_group",
        weight_chunked_mem_mb: int = 1024,
    ):
        return cls(
            type=current_platform.communication_backend,
            alloc_mode=allocation_mode,
            nccl_master_address=gethostip(),
            nccl_master_port=find_free_ports(1)[0],
            nccl_group_name=nccl_group_name,
            weight_chunked_mem_mb=weight_chunked_mem_mb,
        )

    @classmethod
    def from_fsdp_xccl(
        cls,
        allocation_mode: AllocationMode,
        nccl_group_name: str = "update_weight_group",
        weight_chunked_mem_mb: int = 1024,
    ):
        return cls(
            type=current_platform.communication_backend,
            alloc_mode=allocation_mode,
            nccl_master_address=gethostip(),
            nccl_master_port=find_free_ports(1)[0],
            nccl_group_name=nccl_group_name,
            weight_chunked_mem_mb=weight_chunked_mem_mb,
        )


@dataclass
class SaveLoadMeta:
    path: str
    weight_format: str
    with_optim: bool
    tokenizer: Optional[PreTrainedTokenizerFast] = None
    processor: Optional["AutoProcessor"] = None
    base_model_path: str | None = None
    naive_distributed: bool = False


@dataclass
class RolloutStat:
    submitted: int = 0
    accepted: int = 0
    running: int = 0


@dataclass
class StepInfo:
    epoch: int
    epoch_step: int
    global_step: int
    steps_per_epoch: int

    def next(self):
        return StepInfo(
            epoch=self.epoch + (self.epoch_step == self.steps_per_epoch - 1),
            epoch_step=(
                0
                if self.epoch_step == self.steps_per_epoch - 1
                else self.epoch_step + 1
            ),
            global_step=self.global_step + 1,
            steps_per_epoch=self.steps_per_epoch,
        )
