# SPDX-License-Identifier: Apache-2.0

"""Run candidate modules with bounded wall time and parent-owned simulation RPC."""

import importlib.util
import json
import math
import os
import select
import signal
import subprocess
import sys
import time
import types
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from areal.autopilot.elastic.runtime.state.checkpoint import atomic_json


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("_autopilot_candidate", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_source(source: str):
    """Build an explicitly trusted, frozen component without an external source path."""
    import hashlib

    name = "_autopilot_frozen_" + hashlib.sha256(source.encode()).hexdigest()
    module = types.ModuleType(name)
    sys.modules[name] = module
    exec(compile(source, f"<{name}>", "exec"), module.__dict__)
    return module


@dataclass(frozen=True)
class CandidateRunner:
    timeout_seconds: float = 60
    max_output_bytes: int = 2 << 20
    command: tuple[str, ...] = ()

    def __post_init__(self):
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("Candidate timeout must be finite and positive")
        if type(self.max_output_bytes) is not int or self.max_output_bytes < 1:
            raise ValueError("Candidate output limit must be positive")

    def run(
        self,
        source: str,
        request: dict,
        directory: Path,
        simulate: Callable[[dict], dict],
    ) -> dict:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "candidate.py"
        path.write_text(source)
        compile(source, str(path), "exec")
        atomic_json(
            directory / "request.json", {**request, "source_path": str(path.resolve())}
        )
        child_read, parent_write = os.pipe()
        parent_read, child_write = os.pipe()
        os.set_blocking(parent_write, False)
        process = None
        deadline = time.monotonic() + self.timeout_seconds
        result = None
        try:
            with (directory / "worker.log").open("wb") as log:
                process = subprocess.Popen(
                    [
                        *(
                            self.command
                            or (sys.executable, "-m", "areal.autopilot.revision.worker")
                        ),
                        str((directory / "request.json").resolve()),
                        str(child_read),
                        str(child_write),
                    ],
                    pass_fds=(child_read, child_write),
                    stdout=log,
                    stderr=log,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                    cwd=directory,
                    env={
                        **os.environ,
                        "CUDA_VISIBLE_DEVICES": "",
                        "PYTHONDONTWRITEBYTECODE": "1",
                        "PYTHONPATH": str(Path(__file__).resolve().parents[3])
                        + os.pathsep
                        + os.environ.get("PYTHONPATH", ""),
                    },
                )
                os.close(child_read)
                os.close(child_write)
                child_read = child_write = -1
                pending = b""
                while result is None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("Candidate evaluation timed out")
                    if (
                        directory / "worker.log"
                    ).stat().st_size > self.max_output_bytes:
                        raise ValueError("Candidate log exceeds output limit")
                    ready, _, _ = select.select(
                        [parent_read], [], [], min(remaining, 0.1)
                    )
                    if not ready:
                        continue
                    chunk = os.read(parent_read, 65536)
                    if not chunk:
                        raise RuntimeError("Candidate exited without a result")
                    pending += chunk
                    if len(pending) > self.max_output_bytes:
                        raise ValueError("Candidate response exceeds output limit")
                    while b"\n" in pending:
                        line, pending = pending.split(b"\n", 1)
                        message = json.loads(line)
                        if message["type"] == "simulate":
                            try:
                                response = simulate(message)
                            except Exception as error:
                                response = {"error": f"{type(error).__name__}: {error}"}
                            encoded = (
                                json.dumps(response, allow_nan=False) + "\n"
                            ).encode()
                            view = memoryview(encoded)
                            while view:
                                remaining = deadline - time.monotonic()
                                if remaining <= 0:
                                    raise TimeoutError(
                                        "Candidate response channel timed out"
                                    )
                                _, writable, _ = select.select(
                                    [], [parent_write], [], min(remaining, 0.1)
                                )
                                if writable:
                                    try:
                                        written = os.write(parent_write, view)
                                        view = view[written:]
                                    except BlockingIOError:
                                        pass
                        elif message["type"] == "result":
                            result = message
                        else:
                            raise ValueError("Unknown candidate RPC request")
                process.wait(timeout=max(0.01, deadline - time.monotonic()))
                if process.returncode:
                    raise RuntimeError(f"Candidate exited {process.returncode}")
        finally:
            if process is not None:
                # Also reap subprocesses started by the candidate, even after its
                # leader exits normally. This is process cleanup, not a sandbox.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            for descriptor in (child_read, child_write, parent_read, parent_write):
                if descriptor >= 0:
                    os.close(descriptor)
        return result
