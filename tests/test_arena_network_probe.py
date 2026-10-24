# SPDX-License-Identifier: Apache-2.0
"""Normal-path HTTP check of the standalone CPU endpoint, with no live Arena."""

import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def test_network_probe_serves_json_tool_roundtrip_and_sse_without_logging_keys(
    tmp_path,
):
    """Real HTTP and authored Python execution exercise both generation responses."""
    script = (
        Path(__file__).resolve().parents[1]
        / "examples/task_workflows/arena/network_probe_server.py"
    )
    process = subprocess.Popen(
        [
            sys.executable,
            str(script),
            "--bind",
            "127.0.0.1",
            "--port",
            "0",
            "--output",
            str(tmp_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 10
        while not (tmp_path / "server.json").exists():
            assert process.poll() is None and time.monotonic() < deadline
            time.sleep(0.02)
        info = json.loads((tmp_path / "server.json").read_text())
        origin = "http://127.0.0.1:" + str(info["port"])
        http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with http.open(origin + "/healthz") as response:
            assert json.load(response)["probe_id"] == info["probe_id"]
        key = (tmp_path / "key").read_text().strip()

        def post(messages, *, stream):
            request = urllib.request.Request(
                origin + "/v1/chat/completions",
                data=json.dumps(
                    {
                        "model": "scripted-network-probe",
                        "messages": messages,
                        "stream": stream,
                    }
                ).encode(),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer " + key,
                },
            )
            with http.open(request, timeout=10) as response:
                return response.read().decode()

        messages = [{"role": "user", "content": "An authored network fixture."}]
        first = json.loads(post(messages, stream=False))
        message = first["choices"][0]["message"]
        call = message["tool_calls"][0]
        assert call["function"]["name"] == "math_python"
        code = json.loads(call["function"]["arguments"])["code"]
        assert code == "print(6 * 7)"
        tool = subprocess.run(
            [sys.executable, "-I", "-c", code],
            check=True,
            capture_output=True,
            text=True,
        )
        messages += [
            message,
            {
                "role": "tool",
                "tool_call_id": call["id"],
                "content": json.dumps(
                    {"status": "completed", "exit_code": 0, "output": tool.stdout}
                ),
            },
        ]
        data = post(messages, stream=True)
        chunk = json.loads(data.splitlines()[0].removeprefix("data: "))
        assert chunk["choices"][0]["delta"]["content"] == r"\boxed{42}"
        assert data.endswith("data: [DONE]\n\n")
        log = (tmp_path / "requests.jsonl").read_text()
        records = [json.loads(line) for line in log.splitlines()]
        assert [r["completion_id"] for r in records] == [first["id"], chunk["id"]]
        assert [r["tool_result_received"] for r in records] == [False, True]
        assert all(r["synthetic_usage"] and not r["training_eligible"] for r in records)
        assert key not in log and (tmp_path / "key").stat().st_mode & 0o777 == 0o600
    finally:
        process.terminate()
        process.communicate(timeout=5)
