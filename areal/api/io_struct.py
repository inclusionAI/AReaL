# Copyright 2025 Ant Group Inc.
# Licensed under the Apache License, Version 2.0
import enum
import itertools
import os
import re
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple

from PIL.Image import Image as ImageObject
from transformers import AutoProcessor, PreTrainedTokenizerFast

from areal.api.cli_args import GenerationHyperparameters, SaverConfig
from areal.utils.network import find_free_ports, gethostip

if TYPE_CHECKING:
    from areal.api.engine_api import TrainEngine


@dataclass
class LLMRequest:
    rid: str = field(default_factory=lambda: str(uuid.uuid4()))
    input_ids: List[int] = field(default_factory=list)
    gconfig: GenerationHyperparameters = field(
        default_factory=GenerationHyperparameters
    )
    metadata: Dict[str, Any] = field(default_factory=dict)
    model_id: Optional[str] = None


@dataclass
class LLMResponse:
    # outputs
    input_tokens: List[int] = field(default_factory=list)
    output_tokens: List[int] = field(default_factory=list)
    output_logprobs: List[float] = field(default_factory=list)
    output_versions: List[int] = field(default_factory=list)
    stop_reason: Literal["length", "stop", "interrupt"] = "stop"

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
class VLMRequest(LLMRequest):
    image_data: Optional[List[ImageObject | str]] = field(default_factory=list)


@dataclass
class VLMResponse(LLMResponse):
    input_images: List[ImageObject | str] = field(default_factory=list)


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


class AllocationType(enum.Enum):
    COLOCATE = 0
    DECOUPLED_vLLM = 1
    DECOUPLED_SGLANG = 2


@dataclass
class AllocationMode:
    type_: AllocationType
    parallel_strat: Dict[str, Dict[str, int]]

    @property
    def gen_tp_size(self) -> int:
        return self.parallel_strat["gen"]["t"]

    @property
    def gen_pp_size(self) -> int:
        return self.parallel_strat["gen"]["p"]

    @property
    def gen_dp_size(self) -> int:
        return self.parallel_strat["gen"]["d"]

    @property
    def gen_world_size(self) -> int:
        return self.gen_dp_size * self.gen_pp_size * self.gen_tp_size

    @property
    def train_tp_size(self) -> int:
        return self.parallel_strat["*"]["t"]

    @property
    def train_pp_size(self) -> int:
        return self.parallel_strat["*"]["p"]

    @property
    def train_dp_size(self) -> int:
        return self.parallel_strat["*"]["d"]

    @property
    def train_world_size(self) -> int:
        return self.train_dp_size * self.train_pp_size * self.train_tp_size

    @classmethod
    def from_str(cls, allocation_mode: str):
        if "vllm" in allocation_mode:
            alloc_decoupled = AllocationMode.extract_decoupled_alloc(allocation_mode)
            return cls(AllocationType.DECOUPLED_vLLM, alloc_decoupled)
        elif "sglang" in allocation_mode:
            alloc_decoupled = AllocationMode.extract_decoupled_alloc(allocation_mode)
            return cls(AllocationType.DECOUPLED_SGLANG, alloc_decoupled)
        else:
            return cls(
                AllocationType.COLOCATE,
                AllocationMode.extract_parallelism_strategy(allocation_mode),
            )
        raise NotImplementedError(f"Failed to parse allocation: {allocation_mode}")

    @staticmethod
    def extract_parallelism_strategy(allocation_mode: str) -> Dict:
        for x, y, z in itertools.permutations(["d", "t", "p"]):
            pattern = rf"{x}(\d+){y}(\d+){z}(\d+)"
            m = re.match(pattern, allocation_mode)
            if not m:
                continue
            a, b, c = map(int, m.groups())
            # to be consistent with the key-value pattern
            return {
                "*": {
                    x: a,
                    y: b,
                    z: c,
                }
            }
        raise ValueError(
            f"Unknown how to resolve parallelism strategy: {allocation_mode}"
        )

    @staticmethod
    def extract_decoupled_alloc(allocation_mode: str) -> Dict:
        pattern = re.compile(
            r"(?:(?:vllm|sglang)\.(.+?)\+(.+))|(?:(.+?)\+(?:vllm|sglang)\.(.+))"
        )
        m = pattern.match(allocation_mode)
        if not m:
            raise ValueError(
                f"Unknown how to resolve decoupled allocation: {allocation_mode}"
            )
        if m.group(1):
            gen_alloc = m.group(1)
            other_alloc = m.group(2)
        else:
            gen_alloc = m.group(4)
            other_alloc = m.group(3)
        gen_alloc = AllocationMode.extract_parallelism_strategy(gen_alloc)
        other_alloc = AllocationMode.extract_parallelism_strategy(other_alloc)
        other_alloc.update({"gen": gen_alloc["*"]})
        return other_alloc


@dataclass
class ParamSpec:
    name: str
    shape: Tuple
    dtype: str


@dataclass
class WeightUpdateMeta:
    type: Literal["disk", "nccl"]
    path: str | None = None
    alloc_mode: AllocationMode | None = None

    nccl_master_address: str = "127.0.0.1"
    nccl_master_port: int = 29500
    nccl_param_specs: List[ParamSpec] = field(default_factory=list)
    nccl_group_name: str = "update_weight_group"

    @classmethod
    def from_disk(
        cls,
        saver_config: SaverConfig,
    ) -> "WeightUpdateMeta":
        from areal.utils.saver import Saver

        path = os.path.join(
            Saver.get_save_checkpoint_root(saver_config),
            "weight_update",
        )
        return cls(
            type="disk",
            path=path,
        )

    @classmethod
    def from_fsdp_nccl(
        cls,
        allocation_mode: AllocationMode,
        fsdp_engine: "TrainEngine",
        nccl_group_name: str = "update_weight_group",
    ):
        return cls(
            type="nccl",
            alloc_mode=allocation_mode,
            nccl_master_address=gethostip(),
            nccl_master_port=find_free_ports(1)[0],
            nccl_param_specs=fsdp_engine.get_param_specs(),
            nccl_group_name=nccl_group_name,
        )


@dataclass
class SaveLoadMeta:
    path: str
    weight_format: str
    with_optim: bool
    tokenizer: PreTrainedTokenizerFast | None
    processor: AutoProcessor | None
    base_model_path: str | None
    naive_distributed: bool = False


@dataclass
class RolloutStat:
    submitted: int = 0
    accepted: int = 0
    running: int = 0
