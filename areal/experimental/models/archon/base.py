from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from transformers import PretrainedConfig


@dataclass
class BaseModelArgs(ABC):
    """Base class for model arguments."""

    # Attention backend type: "sdpa" or "varlen"
    attn_type: str = "varlen"

    @classmethod
    @abstractmethod
    def from_hf_config(
        cls,
        hf_config: PretrainedConfig,
        is_critic: bool = False,
        **kwargs,
    ) -> BaseModelArgs: ...


class BaseStateDictAdapter(ABC):
    """Base class for HF <-> Archon state dict conversion."""

    def __init__(self, model_config: PretrainedConfig):
        self.model_config = model_config
        self.from_hf_map: dict[str, str | None] = {}
        self.to_hf_map: dict[str, str] = {}

    @abstractmethod
    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]: ...

    @abstractmethod
    def to_hf(self, archon_state_dict: dict[str, Any]) -> dict[str, Any]: ...

    @abstractmethod
    def convert_single_to_hf(
        self, name: str, tensor: torch.Tensor
    ) -> list[tuple[str, torch.Tensor]]: ...


class BaseArchonModel(nn.Module, ABC):
    """Base class for Archon models."""

    model_args: BaseModelArgs
    layers: nn.ModuleDict
    tok_embeddings: nn.Embedding | None
    norm: nn.Module | None
    output: nn.Linear | None
    score: nn.Linear | None

    @abstractmethod
    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor: ...

    @abstractmethod
    def init_weights(self, buffer_device: torch.device | None = None) -> None: ...


__all__ = [
    "BaseModelArgs",
    "BaseStateDictAdapter",
    "BaseArchonModel",
]
