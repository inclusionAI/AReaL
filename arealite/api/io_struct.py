import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

import torch
from gymnasium.core import ActType

from arealite.api.cli_args import GenerationHyperparameters


@dataclass
class LLMServerInfo:
    server_id: str
    host: str
    port: int
    status: str = "healthy"
    last_heartbeat: float = 0
    load: float = 0.0
    version: int = 0


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
    completion: Any
    input_tokens: List[int] = field(default_factory=list)
    output_tokens: List[int] = field(default_factory=list)
    output_logprobs: List[float] = field(default_factory=list)
    output_versions: List[int] = field(default_factory=list)
    stop_reason: Literal["length", "stop", "interrupt"] = "stop"

    # statistics
    latency: float = float("inf")
    ttft: float = float("inf")  # Time to first token
    itl: List[float] = field(default_factory=list)  # List of inter-token latencies


@dataclass
class AgentInferOutput:
    action: ActType
    llm_req: LLMRequest
    llm_resp: LLMResponse


@dataclass
class TrajStats:
    start_time: float = 0.0
    total_reward: float = 0.0
    episode_length: int = 0
    info: Dict = field(default_factory=dict)


@dataclass
class Trajectory:
    data: Dict[str, torch.Tensor]
    stats: TrajStats


@dataclass
class WeightUpdateGroupMeta:
    group_name: str
    ranks: List[int]
    comm_type: str


@dataclass
class WeightMeta:
    group_name: str
    param_name: str
    shape: torch.Size
    dtype: torch.dtype
