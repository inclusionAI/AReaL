#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU-only scripted model endpoint for Arena network checks (Python >=3.6).

No model weights or project dependencies. Returns an authored math_python call,
then a final answer derived from the tool result. Never accepts RL episodes.
"""

import argparse
import hmac
import json
import os
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn


class Server(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def private_json(path, value):
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(value, stream)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    args.output.mkdir(parents=True, exist_ok=True, mode=0o700)
    key_path = args.output / "key"
    if not key_path.exists():
        fd = os.open(str(key_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as stream:
            stream.write(secrets.token_urlsafe(32) + "\n")
    key = key_path.read_text().strip()
    probe_id = secrets.token_hex(8)
    log_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, value, content_type="application/json"):
            data = value if isinstance(value, bytes) else json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/healthz":
                self.reply(200, {"probe_id": probe_id, "synthetic": True})
            elif self.path == "/v1/models":
                self.reply(
                    200,
                    {
                        "object": "list",
                        "data": [{"id": "scripted-network-probe", "object": "model"}],
                    },
                )
            else:
                self.reply(404, {"error": "unknown probe path"})

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self.reply(404, {"error": "unknown probe path"})
                return
            if not hmac.compare_digest(
                self.headers.get("Authorization", ""), "Bearer " + key
            ):
                self.reply(401, {"error": "invalid probe key"})
                return
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if body.get("areal_episode") is not None:
                self.reply(400, {"error": "scripted probe refuses training episodes"})
                return
            results = [m for m in body["messages"] if m["role"] == "tool"]
            if not results:
                message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "probe-tool-" + probe_id,
                            "type": "function",
                            "function": {
                                "name": "math_python",
                                "arguments": json.dumps({"code": "print(6 * 7)"}),
                            },
                        }
                    ],
                }
                finish = "tool_calls"
            else:
                result = json.loads(results[-1]["content"])
                if result.get("status") != "completed" or result.get("exit_code") != 0:
                    message = {
                        "role": "assistant",
                        "content": "Probe tool execution did not complete.",
                    }
                else:
                    message = {
                        "role": "assistant",
                        "content": "\\boxed{" + result["output"].strip() + "}",
                    }
                finish = "stop"
            completion_id = "chatcmpl-probe-" + probe_id + "-" + secrets.token_hex(4)
            response = {
                "id": completion_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "scripted-network-probe",
                "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                "usage": {
                    "prompt_tokens": max(1, len(json.dumps(body)) // 4),
                    "completion_tokens": max(1, len(json.dumps(message)) // 4),
                },
            }
            record = {
                "time": time.time(),
                "probe_id": probe_id,
                "peer": self.client_address[0],
                "completion_id": completion_id,
                "tool_result_received": bool(results),
                "stream": bool(body.get("stream")),
                "synthetic_usage": True,
                "training_eligible": False,
            }
            with log_lock, (args.output / "requests.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            if body.get("stream"):
                delta = dict(message)
                for i, call in enumerate(delta.get("tool_calls", [])):
                    call["index"] = i
                response.update(
                    object="chat.completion.chunk",
                    choices=[{"index": 0, "delta": delta, "finish_reason": finish}],
                )
                data = (
                    "data: " + json.dumps(response) + "\n\ndata: [DONE]\n\n"
                ).encode()
                self.reply(200, data, "text/event-stream")
            else:
                self.reply(200, response)

    server = Server((args.bind, args.port), Handler)
    private_json(
        args.output / "server.json",
        {
            "bind": args.bind,
            "port": server.server_port,
            "probe_id": probe_id,
            "model": "scripted-network-probe",
            "synthetic_usage": True,
            "training_eligible": False,
        },
    )
    print(
        f"Listening on {args.bind}:{server.server_port}; key saved to {key_path}; requests logged without credentials",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
