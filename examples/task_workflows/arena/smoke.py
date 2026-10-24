# SPDX-License-Identifier: Apache-2.0
"""Run real Hermes against a scripted OpenAI server; no model weights or GPU.

The recipe contains authored Python code and a prompt hash, never a gold label.
The final scripted response is derived from the actual math_python output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from examples.task_workflows.arena.runner import run_episode, write_json
from examples.task_workflows.environments.python import PythonSandbox, ToolSettings


@contextmanager
def mock_model(recipe: dict[str, str]):
    requests: list[dict[str, Any]] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            pass

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if not self.path.endswith("/chat/completions"):
                # Hermes can probe local-provider metadata, e.g. /api/show.
                # Such discovery requests are not model generations.
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            requests.append(body)
            messages = body["messages"]
            tool_results = [item for item in messages if item["role"] == "tool"]
            if not tool_results:
                message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "smoke-tool-1",
                            "type": "function",
                            "function": {
                                "name": "math_python",
                                "arguments": json.dumps({"code": recipe["code"]}),
                            },
                        }
                    ],
                }
                finish = "tool_calls"
            else:
                result = json.loads(tool_results[-1]["content"])
                if result.get("status") != "completed" or result.get("exit_code") != 0:
                    message = {
                        "role": "assistant",
                        "content": "Tool infrastructure failure.",
                    }
                else:
                    answer = result["output"].strip()
                    message = {
                        "role": "assistant",
                        "content": "\\boxed{" + answer + "}",
                    }
                finish = "stop"
            usage = {
                "prompt_tokens": max(1, len(json.dumps(body)) // 4),
                "completion_tokens": max(1, len(json.dumps(message)) // 4),
            }
            response = {
                "id": f"chatcmpl-smoke-{len(requests)}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": body["model"],
                "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                "usage": usage,
            }
            if body.get("stream"):
                delta = dict(message)
                for index, call in enumerate(delta.get("tool_calls", [])):
                    call["index"] = index
                chunk = {
                    **response,
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                }
                data = ("data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n").encode()
                content_type = "text/event-stream"
            else:
                data = json.dumps(response).encode()
                content_type = "application/json"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


class TrustedFixturePython(PythonSandbox):
    """Local transport double: only the exact authored recipe may execute.

    This is a portable development check, not a sandbox for model-written code.
    The Arena smoke run always uses ArenaPython/bubblewrap instead.
    """

    def __init__(self, settings: ToolSettings, code: str) -> None:
        super().__init__(settings, "trusted-fixture", image_id="trusted-fixture")
        self.code = code
        self.host_env = {"PATH": os.defpath}
        self.host_process_group = True

    def _start(self) -> None:
        self.container = "trusted-fixture"

    def _command(self, *args: str) -> list[str]:
        if args[-1] != self.code:
            raise RuntimeError("The local fixture cannot execute other code")
        return [sys.executable, *args[1:]]

    def close(self) -> None:
        self.container = None


def smoke(prompt: str, recipe: dict, output: Path, *, local: bool = False) -> dict:
    output = output.resolve()
    if hashlib.sha256(prompt.encode()).hexdigest() != recipe["prompt_sha256"]:
        raise ValueError("Smoke recipe does not match this exact prompt")
    if local and os.environ.get("IS_SANDBOX"):
        raise RuntimeError("The local transport double cannot run on Arena")
    identity = {"incarnation_id": "smoke-not-a-training-session", "trajectory_id": 0}
    runtime = (
        TrustedFixturePython(ToolSettings("fixture", output / "tools"), recipe["code"])
        if local
        else None
    )
    with mock_model(recipe) as (base_url, requests):
        answer = run_episode(
            prompt,
            output=output,
            base_url=base_url,
            api_key="mock-nonsecret-key",
            model="scripted-smoke",
            sampling={"max_tokens": 4096, "temperature": 1.0, "top_p": 1.0},
            episode=identity,
            mock=True,
            runtime=runtime,
        )
    write_json(output / "mock_requests.json", requests)
    if len(requests) != 2 or any(r.get("areal_episode") != identity for r in requests):
        raise RuntimeError(
            "Smoke did not preserve identity across exactly two requests"
        )
    write_json(
        output / "smoke_check.json",
        {
            "status": "OK",
            "native_hermes": "0.19.0",
            "model": "scripted",
            "usage": "synthetic; not training tokens",
            "local_transport_double": local,
            "requests": len(requests),
            "training_eligible": False,
        },
    )
    return answer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query", type=Path, default=os.environ.get("ARENA_QUERY_PATH")
    )
    parser.add_argument(
        "--recipe", type=Path, default=os.environ.get("AUTOPILOT_SMOKE_RECIPE")
    )
    parser.add_argument(
        "--output", type=Path, default=os.environ.get("ARENA_AGENT_OUTPUT_DIR")
    )
    parser.add_argument("--local-trusted-fixture", action="store_true")
    args = parser.parse_args()
    if not all([args.query, args.recipe, args.output]):
        parser.error("query, recipe and output are required")
    smoke(
        args.query.read_text(),
        json.loads(args.recipe.read_text()),
        args.output,
        local=args.local_trusted_fixture,
    )
    if os.environ.get("ARENA_OUTPUT_DIR"):
        write_json(
            Path(os.environ["ARENA_OUTPUT_DIR"]) / "harness_result.json",
            {"status": "OK", "raw": {"model_mode": "mock", "training_eligible": False}},
        )


if __name__ == "__main__":
    main()
