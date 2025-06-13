from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal

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


@dataclass
class Message:
    role: str
    content: Any
    attachment: Any


@dataclass
class LLMRequest:
    rid: str
    model_id: str
    messages: List[Message]
    gconfig: GenerationHyperparameters
    metadata: Dict[str, Any]


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
class Trajectory:
    data: Dict[str, torch.Tensor]
    stats: Dict[str, Any]
