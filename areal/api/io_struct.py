import os
import subprocess
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Optional

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
    input_ids: list[int] = field(default_factory=list)
    gconfig: GenerationHyperparameters = field(
        default_factory=GenerationHyperparameters
    )
    metadata: dict[str, Any] = field(default_factory=dict)
    # tokenizer is used for encode-decode in the inference engine
    tokenizer: PreTrainedTokenizerFast | None = None

    # vlm
    image_data: list[str] | None = field(default_factory=list)
    processor: Optional["AutoProcessor"] = None

    # vlm+vllm:
    vision_msg_vllm: list | None = None

    def copy(self):
        return ModelRequest(
            rid=self.rid,
            input_ids=self.input_ids.copy(),
            gconfig=self.gconfig.new(),
            metadata=self.metadata.copy(),
            tokenizer=self.tokenizer,
            image_data=self.image_data.copy() if self.image_data is not None else None,
            processor=self.processor,
            vision_msg_vllm=(
                self.vision_msg_vllm.copy()
                if self.vision_msg_vllm is not None
                else None
            ),
        )


@dataclass
class ModelResponse:
    # outputs
    input_tokens: list[int] = field(default_factory=list)
    output_tokens: list[int] = field(default_factory=list)
    output_logprobs: list[float] = field(default_factory=list)
    output_versions: list[int] = field(default_factory=list)
    stop_reason: Literal["length", "stop", "tool_calls", "abort"] = "stop"
    # tokenizer is used for encode-decode in the inference engine
    tokenizer: PreTrainedTokenizerFast | None = None

    # vlm
    input_images: list[ImageObject | str] = field(default_factory=list)
    processor: Optional["AutoProcessor"] = None

    # statistics
    latency: float = float("inf")
    ttft: float = float("inf")  # Time to first token
    itl: list[float] = field(default_factory=list)  # List of inter-token latencies

    @property
    def input_len(self) -> int:
        return len(self.input_tokens)

    @property
    def output_len(self) -> int:
        return len(self.output_tokens)

    @property
    def output_tokens_without_stop(self) -> list[int]:
        if self.stop_reason not in ["length", "abort"] and self.output_tokens:
            return self.output_tokens[:-1]
        return self.output_tokens


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
    shape: tuple
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
    lora_name: str = ""
    lora_int_id: int = 0
    base_model_name: str = ""

    clear_checkpoint_after_load: bool = True

    @classmethod
    def from_disk(
        cls,
        experiment_name: str,
        trial_name: str,
        file_root: str,
        name: str = "default",
        use_lora: bool = False,
        clear_checkpoint_after_load: bool = True,
        lora_name: str = "",
        lora_int_id: int = 0,
        base_model_name: str = "",
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
            clear_checkpoint_after_load=clear_checkpoint_after_load,
            lora_name=lora_name,
            lora_int_id=lora_int_id,
            base_model_name=base_model_name,
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
class HttpRequest:
    """Represents an HTTP request to be sent to a remote inference server."""

    endpoint: str
    payload: dict[str, Any]
    method: str = "POST"


@dataclass
class HttpGenerationResult:
    """Parsed result from a generation response."""

    output_tokens: list[int]
    output_logprobs: list[float]
    stop_reason: str


@dataclass
class WeightUpdateRequests:
    """Collection of HTTP requests needed for a weight update operation."""

    requests: list[HttpRequest]


@dataclass
class SaveLoadMeta:
    path: str
    weight_format: str
    with_optim: bool
    tokenizer: PreTrainedTokenizerFast | None = None
    processor: Optional["AutoProcessor"] = None
    base_model_path: str | None = None
    naive_distributed: bool = False


@dataclass
class RolloutStat:
    accepted: int = 0
    enqueued: int = 0
    rejected: int = 0
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


@dataclass
class LocalInfServerInfo:
    """Information about a locally launched inference server."""

    host: str
    port: int
    process: subprocess.Popen
