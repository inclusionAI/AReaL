"""Agent Bridge — HTTP protocol translation layer.

Provides :class:`OpenResponsesBridge` which exposes an OpenAI
Responses-compatible ``POST /v1/responses`` endpoint.  Requests are
translated into DataProxy ``/session/{key}/turn`` calls, routed through
the Router for session affinity.

All inter-service communication is HTTP.

Reference: https://docs.openclaw.ai/gateway/openresponses-http-api
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from areal.utils import logging

from .protocol import generate_run_id

logger = logging.getLogger("AgentBridge")


class AgentBridge(ABC):
    """Base class for HTTP bridge adapters."""

    @abstractmethod
    async def handle_request(self, request: Request) -> Any: ...


class OpenResponsesBridge(AgentBridge):
    """OpenAI Responses API bridge (``POST /v1/responses``).

    Translates between the OpenResponses item-based request format and the
    internal DataProxy turn API.
    """

    def __init__(self, router_addr: str) -> None:
        self._router_addr = router_addr
        self._http = httpx.AsyncClient(timeout=600.0)

    async def handle_request(self, request: Request) -> Any:
        body = await request.json()

        input_items: list[dict[str, Any]] = body.get("input", [])
        instructions: str = body.get("instructions", "")
        model: str = body.get("model", "")
        user: str = body.get("user", "")

        message = self._extract_message(input_items, instructions)
        session_key = self._derive_session_key(user, model)
        run_id = generate_run_id()
        response_id = f"resp-{uuid.uuid4().hex[:12]}"

        metadata = {
            "input": input_items,
            "instructions": instructions,
            "tools": body.get("tools", []),
            "model": model,
            "idempotencyKey": response_id,
            **body.get("metadata", {}),
        }

        try:
            route_resp = await self._http.post(
                f"{self._router_addr}/route",
                json={"session_key": session_key},
            )
            route_resp.raise_for_status()
            data_proxy_addr = route_resp.json()["data_proxy_addr"]

            turn_resp = await self._http.post(
                f"{data_proxy_addr}/session/{session_key}/turn",
                json={
                    "message": message,
                    "run_id": run_id,
                    "queue_mode": "collect",
                    "metadata": metadata,
                },
            )
            turn_resp.raise_for_status()
            result = turn_resp.json()

            output_items: list[dict[str, Any]] = []
            summary = result.get("summary", "")
            if summary:
                output_items.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": summary}],
                    }
                )

            for evt in result.get("events", []):
                if evt.get("type") == "tool_call":
                    output_items.append(
                        {
                            "type": "function_call",
                            "name": evt.get("name", ""),
                            "arguments": evt.get("args", ""),
                        }
                    )

            return JSONResponse(
                {
                    "id": response_id,
                    "object": "response",
                    "status": "completed",
                    "output": output_items,
                    "model": model,
                    "metadata": result.get("metadata", {}),
                }
            )
        except Exception as exc:
            logger.error("OpenResponses request failed: %s", exc)
            return JSONResponse(
                {"error": {"message": str(exc), "type": "server_error"}},
                status_code=500,
            )

    @staticmethod
    def _extract_message(input_items: list[dict[str, Any]], instructions: str) -> str:
        parts: list[str] = []
        if instructions:
            parts.append(instructions)
        for item in input_items:
            if item.get("type") == "message":
                content = item.get("content", "")
                if isinstance(content, list):
                    for block in content:
                        if (
                            isinstance(block, dict)
                            and block.get("type") == "input_text"
                        ):
                            parts.append(block.get("text", ""))
                elif isinstance(content, str):
                    parts.append(content)
            elif item.get("type") == "function_call_output":
                parts.append(f"[tool result] {item.get('output', '')}")
        return "\n".join(parts)

    @staticmethod
    def _derive_session_key(user: str, model: str) -> str:
        if user:
            return f"agent:{model or 'default'}:{user}"
        return f"agent:{model or 'default'}:{uuid.uuid4().hex[:8]}"


def mount_bridge(app: FastAPI, bridge: OpenResponsesBridge) -> None:
    """Mount the OpenResponses bridge on an existing FastAPI app."""

    @app.post("/v1/responses")
    async def responses_endpoint(request: Request):
        return await bridge.handle_request(request)
