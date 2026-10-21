# SPDX-License-Identifier: Apache-2.0

"""Async tool adapter isolating the sandbox SDK from OpenAI Agents dependencies."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any


class SandboxProcess:
    def __init__(self, python: str, *, stderr: Any, ownership_file: Path | None = None):
        self.python, self.stderr = python, stderr
        self.ownership_file = ownership_file
        self.process = None
        self.config = SimpleNamespace(repo_dir="")
        self.instance_id = None
        self.tool_call_failed_count = 0
        self.calls: list[dict[str, Any]] = []
        self._lock = asyncio.Lock()
        self._sequence = 0
        self._broken = False

    async def start(self, data: dict, environment: str) -> None:
        self.process = await asyncio.create_subprocess_exec(
            self.python,
            "-u",
            str(Path(__file__).with_name("sandbox_worker.py")),
            *([str(self.ownership_file)] if self.ownership_file else []),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=self.stderr,
            limit=16 << 20,
        )
        result = await self.request("initialize", data=data, environment=environment)
        self.config.repo_dir = result["repo_dir"]

    async def request(self, method: str, **params) -> Any:
        if self.process is None or self._broken:
            raise RuntimeError("Sandbox RPC is unavailable")
        async with self._lock:
            self._sequence += 1
            request = {"id": self._sequence, "method": method, "params": params}
            try:
                self.process.stdin.write((json.dumps(request) + "\n").encode())
                await self.process.stdin.drain()
                line = await asyncio.wait_for(
                    self.process.stdout.readline(), timeout=1900
                )
                if not line:
                    raise RuntimeError("Sandbox subprocess closed its response stream")
                response = json.loads(line)
                if response.get("id") != self._sequence:
                    raise RuntimeError("Sandbox RPC response does not match request")
                self.instance_id = response.get("instance_id", self.instance_id)
                self.tool_call_failed_count = response.get(
                    "tool_call_failed_count", self.tool_call_failed_count
                )
                if "error" in response:
                    raise RuntimeError(response["error"])
                self.calls.append(
                    {"method": method, "params": params, "result": response["result"]}
                )
                return response["result"]
            except BaseException:
                self._broken = True
                raise

    async def execute_bash(self, command: str) -> dict:
        return await self.request("execute_bash", command=command)

    async def search_replace(self, filepath: str, old_str: str, new_str: str) -> dict:
        return await self.request(
            "search_replace", filepath=filepath, old_str=old_str, new_str=new_str
        )

    async def submit_final_solution(self) -> dict:
        return await self.request("submit_final_solution")

    async def close(self) -> None:
        if self.process is None:
            return
        try:
            if not self._broken and self.process.returncode is None:
                await self.request("release")
        finally:
            if self.process.stdin is not None:
                self.process.stdin.close()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=30)
            except TimeoutError:
                self.process.kill()
                await self.process.wait()
