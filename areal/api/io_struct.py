import enum
import itertools
import os
import re
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
from PIL.Image import Image as ImageObject
from transformers import PreTrainedTokenizerFast

from areal.api.cli_args import GenerationHyperparameters
from areal.utils.network import find_free_ports, gethostip

if TYPE_CHECKING:
    from transformers import AutoProcessor

    from areal.api.engine_api import TrainEngine


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


class AllocationType(enum.Enum):
    COLOCATE = 0
    DECOUPLED_TRAIN = 1
    LLM_SERVER_ONLY = 2
    DECOUPLED_EVAL = 3


@dataclass
class AllocationMode:
    type_: AllocationType
    parallel_strat: Dict[str, Dict[str, int]]
    gen_backend: Optional[str] = None

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
    def gen_instance_size(self):
        return self.gen_tp_size * self.gen_pp_size

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
        # 1. A string like "d2p2t1", representing the N-d parallelism strategy
        para = AllocationMode.extract_parallelism_strategy(allocation_mode)
        if para:
            return cls(
                AllocationType.COLOCATE,
                para,
                AllocationMode.get_gen_backend(allocation_mode),
            )
        para = AllocationMode.extract_decoupled_alloc(allocation_mode)
        # 2. A string like "sglang.d4p1t1+d2pm2", representing a decoupled
        # allocation with 4 GPUs dedicated for SGLang inference and
        # other 4 GPUs dedicated for training
        if "*" in para:
            return cls(
                AllocationType.DECOUPLED_TRAIN,
                para,
                AllocationMode.get_gen_backend(allocation_mode),
            )
        # 3. A string like "sglang.d4p1t1+eval" or "sglang.d4p1t1+cpu",
        # representing a decoupled allocation with 4 GPUs for SGLang server
        # and several CPU client processes to send requests.
        # The number of CPU processes depend on the launcher type.
        # One process for local launcher and `n_gpus_per_node` processes for Ray/SLURM.
        if para:
            return cls(
                AllocationType.DECOUPLED_EVAL,
                para,
                AllocationMode.get_gen_backend(allocation_mode),
            )
        # 4. A string like "sglang.d4p1t1"", representing SGLang server-only allocation.
        para = AllocationMode.extract_gen_alloc(allocation_mode)
        if para:
            return cls(
                AllocationType.LLM_SERVER_ONLY,
                dict(gen=para),
                AllocationMode.get_gen_backend(allocation_mode),
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
        return {}

    @staticmethod
    def extract_gen_alloc(allocation_mode: str) -> Dict:
        pattern = re.compile(r"(?:vllm|sglang)\.(.+?)")
        m = pattern.match(allocation_mode)
        if not m:
            return {}
        return AllocationMode.extract_parallelism_strategy(m.group(1))

    @staticmethod
    def extract_decoupled_alloc(allocation_mode: str) -> Dict:
        pattern = re.compile(
            r"(?:(?:vllm|sglang)\.(.+?)\+(.+))|(?:(.+?)\+(?:vllm|sglang)\.(.+))"
        )
        m = pattern.match(allocation_mode)
        if not m:
            return {}
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

    @staticmethod
    def get_gen_backend(allocation_mode: str) -> Optional[str]:
        if "vllm" in allocation_mode:
            return "vllm"
        if "sglang" in allocation_mode:
            return "sglang"
        return None


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
    nccl_param_specs: List[List[ParamSpec]] = field(default_factory=list)
    nccl_group_name: str = "update_weight_group"

    @classmethod
    def from_disk(
        cls,
        experiment_name: str,
        trial_name: str,
        file_root: str,
        name: str = "default",
    ) -> "WeightUpdateMeta":
        from areal.utils.saver import Saver

        path = os.path.join(
            Saver.get_model_save_root(experiment_name, trial_name, file_root, name),
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
        weight_chunked_mem_mb: int = 1024,
    ):
        param_specs = fsdp_engine.get_param_specs(weight_chunked_mem_mb)
        return cls(
            type="nccl",
            alloc_mode=allocation_mode,
            nccl_master_address=gethostip(),
            nccl_master_port=find_free_ports(1)[0],
            nccl_param_specs=param_specs,
            nccl_group_name=nccl_group_name,
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
