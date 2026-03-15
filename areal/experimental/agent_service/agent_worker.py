"""Agent Worker — stateless HTTP server for agent execution.

The Worker loads an :class:`AgentRunnable` implementation at startup and
exposes a ``POST /run`` endpoint.  Each request is a single turn: the
DataProxy supplies conversation *history* in the request body, and the
Worker returns the agent's response.  The Worker is completely stateless
— session management lives in the DataProxy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from areal.utils import logging
from areal.utils.dynamic_import import import_from_string

from .protocol import QueueMode

logger = logging.getLogger("AgentWorker")


# ---------------------------------------------------------------------------
# Public types — stable API for agent implementations
# ---------------------------------------------------------------------------


@dataclass
class AgentRequest:
    """Structured request passed to the agent.

    Core fields are stable protocol-level attributes.  Framework-specific
    parameters should go in *metadata*.
    """

    message: str
    session_key: str
    run_id: str
    history: list[dict[str, str]] = field(default_factory=list)
    queue_mode: QueueMode = QueueMode.COLLECT
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResponse:
    """Structured result returned by the agent."""

    summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class EventEmitter(Protocol):
    """Callback interface for streaming events from agent to caller."""

    async def emit_delta(self, text: str) -> None: ...
    async def emit_tool_call(self, name: str, args: str) -> None: ...
    async def emit_tool_result(self, name: str, result: str) -> None: ...


@runtime_checkable
class AgentRunnable(Protocol):
    """Minimal protocol for pluggable agent implementations.

    Agent classes are loaded via
    :func:`~areal.utils.dynamic_import.import_from_string` at worker startup.
    The framework handles its own tool execution, memory, and LLM
    interaction — the Agent Service only provides session lifecycle and
    event streaming.

    Reward computation is **not** part of this interface.  Rewards are
    calculated externally by the training pipeline (via reward functions
    applied to exported trajectories), following AReaL's standard RLVR
    pattern.
    """

    async def run(
        self,
        request: AgentRequest,
        *,
        emitter: EventEmitter,
    ) -> AgentResponse: ...


# ---------------------------------------------------------------------------
# In-memory event collector (used by /run endpoint)
# ---------------------------------------------------------------------------


class _CollectingEmitter:
    """Collects events into a list for inclusion in the HTTP response."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def emit_delta(self, text: str) -> None:
        self.events.append({"type": "delta", "text": text})

    async def emit_tool_call(self, name: str, args: str) -> None:
        self.events.append({"type": "tool_call", "name": name, "args": args})

    async def emit_tool_result(self, name: str, result: str) -> None:
        self.events.append({"type": "tool_result", "name": name, "result": result})


# ---------------------------------------------------------------------------
# HTTP server factory
# ---------------------------------------------------------------------------


def create_worker_app(
    agent_cls_path: str,
    **agent_kwargs: Any,
) -> FastAPI:
    """Create the Agent Worker HTTP application.

    Parameters
    ----------
    agent_cls_path : str
        Import path for the agent class (e.g. ``myapp.agent:MyAgent``).
    **agent_kwargs
        Extra keyword arguments forwarded to the agent constructor.
    """
    app = FastAPI(title="AReaL Agent Worker")

    cls = import_from_string(agent_cls_path)
    agent: AgentRunnable = cls(**agent_kwargs)
    if not isinstance(agent, AgentRunnable):
        raise TypeError(
            f"Loaded class {agent_cls_path} does not satisfy AgentRunnable protocol "
            f"(missing async def run(request, *, emitter) method)"
        )
    logger.info("Agent loaded: %s", agent_cls_path)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/run")
    async def run(body: dict[str, Any]):
        request = AgentRequest(
            message=body.get("message", ""),
            session_key=body.get("session_key", ""),
            run_id=body.get("run_id", ""),
            history=body.get("history", []),
            queue_mode=QueueMode(body.get("queue_mode", "collect")),
            metadata=body.get("metadata", {}),
        )

        emitter = _CollectingEmitter()

        try:
            response = await agent.run(request, emitter=emitter)
        except Exception as exc:
            logger.exception("Agent run failed (session=%s)", request.session_key)
            return JSONResponse(
                {"error": {"message": str(exc), "type": type(exc).__name__}},
                status_code=500,
            )

        return {
            "summary": response.summary,
            "metadata": response.metadata,
            "events": emitter.events,
        }

    return app


def main() -> None:
    import argparse
    import asyncio
    import threading

    import httpx
    import uvicorn

    from .data_proxy import create_data_proxy_app

    parser = argparse.ArgumentParser(description="Agent Worker + DataProxy")
    parser.add_argument("--agent", required=True, help="Agent import path")
    parser.add_argument("--router-addr", required=True, help="Router HTTP address")
    parser.add_argument("--worker-port", type=int, default=9000)
    parser.add_argument("--proxy-port", type=int, default=9100)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    worker_addr = f"http://{args.host}:{args.worker_port}"
    proxy_addr = f"http://{args.host}:{args.proxy_port}"

    worker_app = create_worker_app(args.agent)
    proxy_app = create_data_proxy_app(worker_addr=worker_addr)

    def run_worker():
        uvicorn.run(worker_app, host=args.host, port=args.worker_port, log_level="info")

    threading.Thread(target=run_worker, daemon=True).start()

    async def register():
        async with httpx.AsyncClient() as client:
            await client.post(f"{args.router_addr}/register", json={"addr": proxy_addr})

    asyncio.run(register())
    uvicorn.run(proxy_app, host=args.host, port=args.proxy_port, log_level="info")


if __name__ == "__main__":
    main()
