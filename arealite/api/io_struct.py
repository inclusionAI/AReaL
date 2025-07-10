# Copyright 2025 Ant Group Inc.
# Licensed under the Apache License, Version 2.0

import enum
import itertools
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from transformers import PreTrainedTokenizerFast

from arealite.api.cli_args import GenerationHyperparameters


@dataclass
class LLMRequest:
    rid: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: Optional[str] = None
    input_ids: List[int] = field(default_factory=list)
    gconfig: GenerationHyperparameters = field(
        default_factory=GenerationHyperparameters
    )
    metadata: Dict[str, Any] = field(default_factory=dict)
    model_id: Optional[str] = None


@dataclass
class LLMResponse:
    # outputs
    completions: str
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
    DECOUPLED_vLLM = 1
    DECOUPLED_SGLANG = 2


@dataclass
class AllocationMode:
    type_: AllocationType
    parallel_strat: None | Dict[str, Dict[str, int]]

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
        alloc_decoupled = AllocationMode.extract_decoupled_alloc(allocation_mode)
        if "vllm" in allocation_mode:
            return cls(AllocationType.DECOUPLED_vLLM, alloc_decoupled)
        elif "sglang" in allocation_mode:
            return cls(AllocationType.DECOUPLED_SGLANG, alloc_decoupled)
        raise NotImplementedError(f"Failed to parse allocation: {allocation_mode}")

    @staticmethod
    def extract_3d_alloc(allocation_mode: str) -> Dict | None:
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

    @staticmethod
    def extract_decoupled_alloc(allocation_mode: str) -> Dict | None:
        pattern = re.compile(
            r"(?:(?:vllm|sglang)\.(.+?)\+(.+))|(?:(.+?)\+(?:vllm|sglang)\.(.+))"
        )
        m = pattern.match(allocation_mode)
        if not m:
            return
        if m.group(1):
            gen_alloc = m.group(1)
            other_alloc = m.group(2)
        else:
            gen_alloc = m.group(4)
            other_alloc = m.group(3)
        gen_alloc = AllocationMode.extract_3d_alloc(gen_alloc)
        if not gen_alloc:
            return
        other_alloc = AllocationMode.extract_3d_alloc(
            other_alloc
        ) or AllocationMode.extract_key_value_alloc(other_alloc)
        if not other_alloc:
            return
        other_alloc.update({"gen": gen_alloc["*"]})
        return other_alloc


@dataclass
class WeightUpdateMeta:
    type: str
    path: str | None
    alloc_mode: AllocationMode | None
    comm_backend: str | None
    model_version: int = 0


@dataclass
class SaveLoadMeta:
    path: str
    weight_format: str
    with_optim: bool
    tokenizer: PreTrainedTokenizerFast | None
    base_model_path: str | None


@dataclass
class RolloutStat:
    submitted: int = 0
    accepted: int = 0
    running: int = 0
