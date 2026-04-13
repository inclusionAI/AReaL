# SPDX-License-Identifier: Apache-2.0

"""AReaL Agent Service — agent-level inference tier.

Exposes complete agent sessions (autonomous multi-step reasoning, tool use,
memory) via independent HTTP microservices: Gateway, Router, DataProxy,
and Worker.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

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
from .types import AgentRequest, AgentResponse, AgentRunnable, EventEmitter

if TYPE_CHECKING:
    from .data_proxy import DataProxyClient, create_data_proxy_app
    from .gateway import OpenResponsesBridge, create_gateway_app, mount_bridge
    from .router import RouterClient, create_router_app
    from .worker import create_worker_app

_LAZY_IMPORTS: dict[str, str] = {
    "DataProxyClient": ".data_proxy",
    "OpenResponsesBridge": ".gateway",
    "RouterClient": ".router",
    "create_data_proxy_app": ".data_proxy",
    "create_gateway_app": ".gateway",
    "create_router_app": ".router",
    "create_worker_app": ".worker",
    "mount_bridge": ".gateway",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module = importlib.import_module(_LAZY_IMPORTS[name], __package__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AgentRequest",
    "AgentResponse",
    "AgentRunnable",
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
