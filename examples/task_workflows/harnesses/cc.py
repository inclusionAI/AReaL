# SPDX-License-Identifier: Apache-2.0

"""Claude Code's SDK loop with task-local MCP tools and AReaL inference routing."""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import time
from dataclasses import fields, is_dataclass
from typing import Any
from urllib.parse import urlsplit

from aiohttp import ClientSession, ClientTimeout, web

from areal.v2.agent_service.types import AgentResponse


def _serialize(value: Any) -> Any:
    if is_dataclass(value):
        return {
            "type": type(value).__name__,
            **{
                item.name: _serialize(getattr(value, item.name))
                for item in fields(value)
            },
        }
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_serialize(item) for item in value]
    return value


class InferenceRelay:
    """Bind every CLI HTTP generation (also compaction) to one inference episode.

    A loopback relay is needed because the closed CLI cannot attach AReaL's JSON
    identity itself. It never changes process-wide environment variables.
    """

    def __init__(self, metadata: dict[str, Any]) -> None:
        upstream = metadata.get("areal_inference") or {}
        self.base_url = (
            str(upstream.get("base_url", "")).rstrip("/").removesuffix("/v1")
        )
        self.api_key = str(upstream.get("api_key", ""))
        self.model = str(upstream.get("model", ""))
        self.episode = upstream.get("episode")
        if (
            urlsplit(self.base_url).scheme not in ("http", "https")
            or not self.api_key
            or not self.model
        ):
            raise ValueError(
                "Claude Code tasks require explicit AReaL inference URL, key and model"
            )
        self.key = secrets.token_urlsafe(32)
        self.calls: list[dict[str, Any]] = []
        self._runner = None
        self._http = None

    async def __aenter__(self):
        app = web.Application(client_max_size=64 * 1024 * 1024)
        app.router.add_post("/v1/messages", self._messages)
        self._http = ClientSession(timeout=ClientTimeout(total=None, sock_connect=30))
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        self.url = "http://127.0.0.1:" + str(self._runner.addresses[0][1])
        return self

    async def __aexit__(self, *args):
        try:
            await self._runner.cleanup()
        finally:
            await self._http.close()

    async def _messages(self, request: web.Request) -> web.StreamResponse:
        token = request.headers.get("x-api-key") or request.headers.get(
            "authorization", ""
        ).removeprefix("Bearer ")
        if not secrets.compare_digest(token, self.key):
            raise web.HTTPUnauthorized()
        body = await request.json()
        if body.get("areal_episode") not in (None, self.episode):
            raise web.HTTPConflict(text="Inference episode identity mismatch")
        body["model"] = self.model
        if self.episode is not None:
            body["areal_episode"] = self.episode
        body.update(
            max_tokens=int(os.environ.get("AREAL_TASK_MAX_TOKENS", "4096")),
            temperature=float(os.environ.get("AREAL_TASK_TEMPERATURE", "1.0")),
            top_p=float(os.environ.get("AREAL_TASK_TOP_P", "1.0")),
        )
        clock = time.perf_counter()
        record = {
            "started": time.time(),
            "finished": None,
            "seconds": None,
            "model": self.model,
            "status": None,
            "timing_scope": "relay_receive_to_response_end",
        }
        self.calls.append(record)
        try:
            async with self._http.post(
                self.base_url + "/v1/messages",
                json=body,
                headers={
                    "Authorization": "Bearer " + self.api_key,
                    "anthropic-version": "2023-06-01",
                },
            ) as upstream:
                record["status"] = upstream.status
                response = web.StreamResponse(
                    status=upstream.status,
                    headers={
                        "Content-Type": upstream.headers.get(
                            "Content-Type", "application/json"
                        )
                    },
                )
                await response.prepare(request)
                async for chunk in upstream.content.iter_any():
                    await response.write(chunk)
                await response.write_eof()
                return response
        finally:
            record.update(finished=time.time(), seconds=time.perf_counter() - clock)


class ClaudeCodeHarness:
    """Native SDK orchestration; benchmark preparation and grading stay outside."""

    async def run(self, request: Any, *, runtime: Any, emitter: Any) -> AgentResponse:
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            ClaudeSDKClient,
            ResultMessage,
            create_sdk_mcp_server,
            tool,
        )

        schemas = runtime.tool_schemas
        tools = []
        for schema in schemas:

            def make_handler(name):
                async def invoke(arguments):
                    await emitter.emit_tool_call(name=name, args=json.dumps(arguments))
                    result = await asyncio.to_thread(runtime.invoke, name, arguments)
                    await emitter.emit_tool_result(name=name, result=result)
                    return {"content": [{"type": "text", "text": result}]}

                return invoke

            tools.append(
                tool(schema["name"], schema["description"], schema["parameters"])(
                    make_handler(schema["name"])
                )
            )
        server = create_sdk_mcp_server("task", tools=tools)
        directory = runtime.directory / "cc"
        await asyncio.to_thread(directory.mkdir, parents=True)
        transcript = [
            {
                "sent": time.time(),
                "message": {"type": "UserMessage", "content": request.message},
            }
        ]
        final = None
        async with InferenceRelay(request.metadata) as relay:
            env = {
                "ANTHROPIC_BASE_URL": relay.url,
                "ANTHROPIC_API_KEY": relay.key,
                "ANTHROPIC_AUTH_TOKEN": "",
                "ANTHROPIC_CUSTOM_HEADERS": "",
                "CLAUDE_CODE_OAUTH_TOKEN": "",
                "ANTHROPIC_MODEL": relay.model,
                "ANTHROPIC_DEFAULT_OPUS_MODEL": relay.model,
                "ANTHROPIC_DEFAULT_SONNET_MODEL": relay.model,
                "ANTHROPIC_DEFAULT_HAIKU_MODEL": relay.model,
                "CLAUDE_CODE_SUBAGENT_MODEL": relay.model,
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                "CLAUDE_CODE_USE_BEDROCK": "0",
                "CLAUDE_CODE_USE_VERTEX": "0",
                "CLAUDE_CODE_USE_FOUNDRY": "0",
                "CLAUDE_CONFIG_DIR": str(directory / "config"),
                "NO_PROXY": "127.0.0.1,localhost,::1",
                "MCP_TOOL_TIMEOUT": str(int((runtime.settings.timeout + 180) * 1000)),
            }
            options = ClaudeAgentOptions(
                model=relay.model,
                env=env,
                cwd=str(directory),
                tools=[],
                allowed_tools=["mcp__task__" + schema["name"] for schema in schemas],
                mcp_servers={"task": server},
                setting_sources=[],
                system_prompt={
                    "type": "preset",
                    "preset": "claude_code",
                    "append": "All task files and execution are in the provided MCP task environment. "
                    "Use those tools for all inspection, edits and tests; local paths are not task paths.",
                },
                extra_args={"strict-mcp-config": None, "no-session-persistence": None},
            )
            if os.environ.get("AREAL_CC_CLI_PATH"):
                options.cli_path = os.environ["AREAL_CC_CLI_PATH"]
            if os.environ.get("AREAL_CC_MAX_TURNS"):
                options.max_turns = int(os.environ["AREAL_CC_MAX_TURNS"])

            async def prompt():
                yield {
                    "type": "user",
                    "message": {"role": "user", "content": request.message},
                }

            try:
                async with ClaudeSDKClient(options=options) as client:
                    await client.query(prompt())
                    async for message in client.receive_response():
                        transcript.append(
                            {"received": time.time(), "message": _serialize(message)}
                        )
                        if isinstance(message, AssistantMessage) and getattr(
                            message, "error", None
                        ):
                            raise RuntimeError(
                                "Claude Code model request failed: "
                                + str(message.error)
                            )
                        if isinstance(message, ResultMessage):
                            final = message
                if final is None:
                    raise RuntimeError("Claude Code ended without a result")
                if final.is_error and final.subtype not in (
                    "error_max_turns",
                    "error_max_budget_usd",
                ):
                    raise RuntimeError("Claude Code execution failed: " + final.subtype)
                answer = final.result or ""
                if not answer:
                    for item in reversed(transcript):
                        message = item["message"]
                        if message["type"] == "AssistantMessage":
                            answer = "".join(
                                block["text"]
                                for block in message["content"]
                                if block["type"] == "TextBlock"
                            )
                            break
                if answer:
                    await emitter.emit_delta(answer)
                return AgentResponse(
                    summary=answer,
                    metadata={
                        "harness": "cc",
                        "completed": not final.is_error
                        and final.stop_reason != "max_tokens",
                        "api_calls": len(relay.calls),
                        "inference_calls": relay.calls,
                        "transcript": transcript,
                        "cc_result": _serialize(final),
                    },
                )
            finally:
                await asyncio.to_thread(
                    (directory / "transcript.json").write_text,
                    json.dumps(
                        {"messages": transcript, "inference_calls": relay.calls},
                        ensure_ascii=False,
                        indent=2,
                    ),
                )
