"""AReaL Agent Service — agent-level inference tier.

Exposes complete agent sessions (autonomous multi-step reasoning, tool use,
memory) via independent HTTP microservices: Gateway, Router, DataProxy,
and Worker.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from .config import AgentServiceConfig
from .protocol import (
    EventFrame,
    Frame,
    FrameType,
    QueueMode,
    RequestFrame,
    RequestMethod,
    ResponseFrame,
    RunStatus,
    generate_run_id,
    make_complete_response,
    make_delta_event,
    make_failed_response,
    make_tool_call_event,
    parse_frame,
    serialize_frame,
)

if TYPE_CHECKING:
    from .agent_bridge import AgentBridge, OpenResponsesBridge, mount_bridge
    from .agent_gateway import create_gateway_app
    from .agent_router import RouterClient, create_router_app
    from .agent_worker import (
        AgentRequest,
        AgentResponse,
        AgentRunnable,
        EventEmitter,
        create_worker_app,
    )
    from .data_proxy import DataProxyClient, create_data_proxy_app

_LAZY_IMPORTS: dict[str, str] = {
    "AgentBridge": ".agent_bridge",
    "AgentRequest": ".agent_worker",
    "AgentResponse": ".agent_worker",
    "AgentRunnable": ".agent_worker",
    "DataProxyClient": ".data_proxy",
    "EventEmitter": ".agent_worker",
    "OpenResponsesBridge": ".agent_bridge",
    "RouterClient": ".agent_router",
    "create_data_proxy_app": ".data_proxy",
    "create_gateway_app": ".agent_gateway",
    "create_router_app": ".agent_router",
    "create_worker_app": ".agent_worker",
    "mount_bridge": ".agent_bridge",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module = importlib.import_module(_LAZY_IMPORTS[name], __package__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AgentBridge",
    "AgentRequest",
    "AgentResponse",
    "AgentRunnable",
    "AgentServiceConfig",
    "DataProxyClient",
    "EventEmitter",
    "EventFrame",
    "Frame",
    "FrameType",
    "OpenResponsesBridge",
    "QueueMode",
    "RequestFrame",
    "RequestMethod",
    "ResponseFrame",
    "RouterClient",
    "RunStatus",
    "create_data_proxy_app",
    "create_gateway_app",
    "create_router_app",
    "create_worker_app",
    "generate_run_id",
    "make_complete_response",
    "make_delta_event",
    "make_failed_response",
    "make_tool_call_event",
    "mount_bridge",
    "parse_frame",
    "serialize_frame",
]
