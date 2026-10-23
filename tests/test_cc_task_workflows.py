# SPDX-License-Identifier: Apache-2.0

"""Normal paths with the real Claude SDK/CLI; model and Docker transport are doubles."""

import asyncio
import json
import os

import httpx
import pytest
from aiohttp import ClientSession, web

from examples.task_workflows.benchmarks.dapo_math import DapoBenchmark, MathTask
from examples.task_workflows.benchmarks.swe_rebench import (
    SWERebenchBenchmark,
    SWERebenchTask,
)
from examples.task_workflows.client import ClientSettings, EpisodeClient
from examples.task_workflows.harnesses.cc import InferenceRelay
from examples.task_workflows.service import CCTaskAgent
from tests import test_task_workflows as task_fixtures

from areal.v2.agent_service.types import AgentRequest

repository_case = task_fixtures.repository_case
docker_transport = task_fixtures.docker_transport


def _model_response(body, block, stop_reason):
    """Anthropic wire response consumed by the unmodified bundled Claude Code CLI."""
    message = {
        "id": "msg_offline",
        "type": "message",
        "role": "assistant",
        "model": body["model"],
        "content": [block],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 100, "output_tokens": 20},
    }
    if not body.get("stream"):
        return web.json_response(message)
    start = {
        **message,
        "content": [],
        "stop_reason": None,
        "usage": {"input_tokens": 100, "output_tokens": 0},
    }
    if block["type"] == "tool_use":
        initial = {**block, "input": {}}
        delta = {"type": "input_json_delta", "partial_json": json.dumps(block["input"])}
    else:
        initial = {"type": "text", "text": ""}
        delta = {"type": "text_delta", "text": block["text"]}
    events = [
        {"type": "message_start", "message": start},
        {"type": "content_block_start", "index": 0, "content_block": initial},
        {"type": "content_block_delta", "index": 0, "delta": delta},
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": 20},
        },
        {"type": "message_stop"},
    ]
    return web.Response(
        body="".join(
            "event: " + event["type"] + "\ndata: " + json.dumps(event) + "\n\n"
            for event in events
        ).encode(),
        content_type="text/event-stream",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("benchmark_name", ["dapo_math", "swe_rebench"])
async def test_cc_native_sdk_uses_shared_environment_grading_and_reward(
    tmp_path, monkeypatch, docker_transport, repository_case, benchmark_name
):
    """Actual CC executes an MCP tool, calls the model again and submits the grade."""
    pytest.importorskip(
        "claude_agent_sdk", reason="Requires the project's declared Claude SDK"
    )
    from areal.experimental.openai.anthropic import translate_anthropic_request

    if benchmark_name == "swe_rebench" and not os.environ.get(
        "AREAL_SWE_EVALUATOR_ROOT"
    ):
        pytest.skip("Set AREAL_SWE_EVALUATOR_ROOT to verify real upstream SWE grading")
    monkeypatch.setenv("AREAL_TASK_TOOL_ROOT", str(tmp_path / "tools"))
    monkeypatch.setenv("AREAL_TASK_PYTHON_IMAGE", "offline-image")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "parent-key-must-not-be-used")
    benchmark = (
        DapoBenchmark() if benchmark_name == "dapo_math" else SWERebenchBenchmark()
    )
    task = (
        MathTask("math", "Compute six times seven.", "42")
        if benchmark_name == "dapo_math"
        else SWERebenchTask(repository_case)
    )
    identity = {"incarnation_id": "source-cc", "trajectory_id": 3}
    seen = []

    async def model(request):
        body = await request.json()
        assert request.headers["Authorization"] == "Bearer per-episode-key"
        assert body["areal_episode"] == identity and body["model"] == "local-policy"
        translated = translate_anthropic_request(body)
        assert translated["model"] == "local-policy" and translated["messages"]
        seen.append(body)
        results = [
            block
            for message in body["messages"]
            if isinstance(message.get("content"), list)
            for block in message["content"]
            if block.get("type") == "tool_result"
        ]
        if not results:
            name = "math_python" if benchmark_name == "dapo_math" else "workspace_exec"
            arguments = (
                {"code": "print(6 * 7)"}
                if benchmark_name == "dapo_math"
                else {
                    "command": "printf 'def add(a, b):\\n    return a + b\\n' > calc.py"
                }
            )
            assert any(tool["name"] == "mcp__task__" + name for tool in body["tools"])
            return _model_response(
                body,
                {
                    "type": "tool_use",
                    "id": "toolu_offline",
                    "name": "mcp__task__" + name,
                    "input": arguments,
                },
                "tool_use",
            )
        return _model_response(
            body, {"type": "text", "text": r"\boxed{42}"}, "end_turn"
        )

    app = web.Application()
    app.router.add_post("/v1/messages", model)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    inference_url = "http://127.0.0.1:" + str(runner.addresses[0][1])
    agent = CCTaskAgent()
    responses = []

    class Emitter:
        def __init__(self):
            self.text = []

        async def emit_delta(self, text):
            self.text.append(text)

        async def emit_tool_call(self, **kwargs):
            pass

        async def emit_tool_result(self, **kwargs):
            pass

    async def control(request):
        body = json.loads(request.content)
        if request.url.path == "/rl/start_session":
            return httpx.Response(
                201,
                json={
                    "sessions": [
                        {
                            "session_id": "cc-session",
                            "session_api_key": "per-episode-key",
                            "areal_episode": identity,
                        }
                    ]
                },
            )
        if request.url.path == "/v1/responses":
            emitter = Emitter()
            response = await agent.run(
                AgentRequest(
                    body["input"][0]["content"],
                    body["user"],
                    "run",
                    metadata={
                        **body["metadata"],
                        "areal_inference": {
                            "base_url": body["inf_base_url"],
                            "model": body["inf_model"],
                            "api_key": body["session_api_key"],
                            "episode": body["inf_episode"],
                        },
                    },
                ),
                emitter=emitter,
            )
            responses.append(response)
            return httpx.Response(
                200,
                json={
                    "status": "completed",
                    "metadata": response.metadata,
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {"type": "output_text", "text": "".join(emitter.text)}
                            ],
                        }
                    ],
                },
            )
        if request.url.path == "/rl/set_reward":
            assert body == {"reward": 1.0, "areal_episode": identity}
            return httpx.Response(
                200, json={"trajectory_id": 3, "trajectory_ready": True}
            )
        assert request.url.path == "/sessions/close"
        await agent.close_session(body["session_key"])
        return httpx.Response(200, json={"status": "closed"})

    output = tmp_path / "results"
    output.mkdir()
    settings = ClientSettings(
        "http://agent",
        inference_url,
        "agent-admin",
        "inference-admin",
        "local-policy",
        "resident",
        output,
    )
    try:
        async with httpx.AsyncClient(transport=httpx.MockTransport(control)) as http:
            summary = await asyncio.wait_for(
                EpisodeClient(settings, http, benchmark).run([task]), 60
            )
        rows = [
            json.loads(line)
            for line in (output / "episodes.jsonl").read_text().splitlines()
        ]
        assert summary["infrastructure_errors"] == 0, rows
        assert summary["completed"] == summary["correct"] == 1
        assert len(seen) == summary["model_calls"] == 2
        assert summary["tool_executions"] == 1
        response = responses[0]
        assert response.metadata["harness"] == "cc"
        assert response.metadata["completed"] is True
        assert (
            response.metadata["transcript"][0]["message"]["content"]
            == benchmark.prepare(task).message
        )
        assert all(
            call["seconds"] >= 0 and call["finished"] >= call["started"]
            for call in response.metadata["inference_calls"]
        )
        assert rows[0]["timing"]["tool_seconds"] > 0
        assert "ToolUseBlock" in json.dumps(response.metadata["transcript"])
        assert "ToolResultBlock" in json.dumps(response.metadata["transcript"])
        assert "per-episode-key" not in json.dumps(rows)
        assert os.environ["ANTHROPIC_API_KEY"] == "parent-key-must-not-be-used"
        assert not list(docker_transport.iterdir())
    finally:
        await agent.close_all_sessions()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_cc_relays_isolate_concurrent_inference_sessions():
    """Concurrent episodes retain their own key, model and JSON identity."""
    observed = []

    async def model(request):
        body = await request.json()
        observed.append((request.headers["Authorization"], body))
        return web.json_response({"content": []})

    app = web.Application()
    app.router.add_post("/v1/messages", model)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    base = "http://127.0.0.1:" + str(runner.addresses[0][1])

    async def one(index):
        metadata = {
            "areal_inference": {
                "base_url": base + "/v1",
                "api_key": f"key-{index}",
                "model": f"model-{index}",
                "episode": {
                    "incarnation_id": f"source-{index}",
                    "trajectory_id": index,
                },
            }
        }
        async with InferenceRelay(metadata) as relay, ClientSession() as http:
            async with http.post(
                relay.url + "/v1/messages",
                headers={"x-api-key": relay.key},
                json={"model": "claude-alias", "messages": []},
            ) as response:
                assert response.status == 200
                await response.read()
        assert len(relay.calls) == 1 and relay.calls[0]["seconds"] >= 0

    try:
        await asyncio.gather(one(1), one(2))
        for key, body in observed:
            index = int(key.rsplit("-", 1)[1])
            assert body["model"] == f"model-{index}"
            assert body["areal_episode"] == {
                "incarnation_id": f"source-{index}",
                "trajectory_id": index,
            }
        assert len(observed) == 2
    finally:
        await runner.cleanup()
