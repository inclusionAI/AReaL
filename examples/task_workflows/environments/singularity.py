# SPDX-License-Identifier: Apache-2.0
"""Opt-in host-side Apptainer tools. The agent image only needs the socket client.

Run ``serve`` on the compute host, using Python 3.10+ and its singularity command.
No model, inference key, or training dependency is needed by the host process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import socketserver
import threading
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from examples.task_workflows.environments.python import (
    PythonSandbox,
    ToolSettings,
    run_limited,
)
from examples.task_workflows.environments.repository import (
    RepositoryEnvironment,
    RepositorySettings,
)
from examples.task_workflows.trace import mark_tool_execution

_MAX_MESSAGE = 128 * 1024 * 1024


def rpc(action: str, **payload: Any) -> Any:
    """A private same-user Unix socket, mounted only into the agent/client image."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(
            max(
                float(os.environ.get("AREAL_TASK_HOST_TIMEOUT", "900")),
                float(payload.get("timeout", 0)) + 120,
            )
        )
        connection.connect(os.environ["AREAL_TASK_HOST_SOCKET"])
        connection.sendall(json.dumps({"action": action, **payload}).encode() + b"\n")
        with connection.makefile("rb") as stream:
            response = _read(stream)
    if "error" in response:
        raise RuntimeError("Singularity host: " + response["error"])
    return response["result"]


def _read(stream: Any) -> dict[str, Any]:
    row = stream.readline(_MAX_MESSAGE + 1)
    if not row.endswith(b"\n") or len(row) > _MAX_MESSAGE:
        raise ValueError("Incomplete or oversized host message")
    return json.loads(row)


def create_environment(
    provider: Any, settings: Any, task_id: str, *, image_id: str, **options: Any
) -> Any:
    backend = os.environ.get("AREAL_TASK_RUNTIME", "docker")
    if backend == "docker":
        return provider(settings, task_id, image_id=image_id, **options)
    if backend != "singularity":
        raise ValueError(f"Unknown task runtime: {backend}")
    return RemoteEnvironment(provider, settings, task_id, image_id, options)


class RemoteEnvironment:
    """The existing harness contract; no container executable runs in this process."""

    def __init__(
        self,
        provider: Any,
        settings: Any,
        task_id: str,
        image_id: str,
        options: dict[str, Any],
    ) -> None:
        self.kind, self.toolset = provider.kind, provider.toolset
        self.tool_schemas = provider.tool_schemas
        self.settings, self.task_id = settings, task_id
        self.directory = settings.root / task_id
        self.directory.mkdir(parents=True, exist_ok=False)
        self._closed, self._metadata = False, {}
        rpc(
            "open",
            kind=self.kind,
            task_id=task_id,
            image=image_id,
            settings={**asdict(settings), "root": str(settings.root)},
            options=options,
        )

    def execute(self, value: str) -> str:
        return self.invoke(
            "math_python" if self.kind == "python" else "workspace_exec",
            {"code" if self.kind == "python" else "command": value},
        )

    def invoke(self, name: str, arguments: dict[str, Any]) -> str:
        result = rpc("invoke", task_id=self.task_id, name=name, arguments=arguments)
        timing = result["timing"]
        if timing:
            mark_tool_execution(
                timing["started"], seconds=timing["finished"] - timing["started"]
            )
        return result["value"]

    def export_artifacts(self) -> dict[str, Any]:
        return rpc("export", task_id=self.task_id)

    def metadata(self) -> dict[str, Any]:
        return self._metadata if self._closed else rpc("metadata", task_id=self.task_id)

    def close(self) -> None:
        if not self._closed:
            self._metadata = rpc("close", task_id=self.task_id)
            self._closed = True


class _SIFMixin:
    """One serialized overlay per task; every call is launched by the host runtime."""

    host_process_group = True

    def __init__(
        self, *args: Any, sif: Path, executable: str, overlay_mb: int, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.sif, self.executable, self.overlay_mb = sif, executable, overlay_mb
        self.overlay = self.directory / "overlay.img"
        self.scratch = self.directory / "scratch"
        self.patches: Path | None = None
        # Do not inherit automatic mounts or *_ENV injections from the training job.
        self.host_env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("SINGULARITY_", "APPTAINER_"))
        }

    def _launch(self, *program: str, pwd: str = "/") -> list[str]:
        command = [
            self.executable,
            "exec",
            "--fakeroot",
            "--containall",
            "--cleanenv",
            "--no-home",
            "--no-mount",
            "hostfs,bind-paths,cwd",
            "--net",
            "--network",
            "none",
            "--overlay",
            str(self.overlay),
            "--workdir",
            str(self.scratch),
            "--pwd",
            pwd,
            "--env",
            "PYTHONDONTWRITEBYTECODE=1,CUDA_VISIBLE_DEVICES=,_JAVA_OPTIONS=-Djava.net.preferIPv6Addresses=false",
        ]
        if self.patches:
            command += ["--bind", str(self.patches) + ":/patches:ro"]
        return [*command, str(self.sif), *program]

    def _command(self, *program: str) -> list[str]:
        return self._launch(
            *program, pwd=self.workdir if self.kind == "repository" else "/work"
        )

    def _control(self, *args: str) -> str:
        if args[0] != "exec":
            raise ValueError("Only container exec is supported by the SIF adapter")
        return self._checked(self._command(*args[2:]))

    def _checked(self, command: list[str]) -> str:
        result = run_limited(
            command,
            env=self.host_env,
            timeout=60,
            output_bytes=1024 * 1024,
            process_group=True,
        )
        if result["status"] != "completed" or result["exit_code"] != 0:
            raise RuntimeError(
                "Singularity initialization/control failed: " + result["output"]
            )
        return result["output"].strip()

    def _start(self) -> None:
        if self.container is not None:
            return
        self.scratch.mkdir(exist_ok=True)
        self._checked(
            [
                self.executable,
                "overlay",
                "create",
                "--fakeroot",
                "--sparse",
                "--size",
                str(self.overlay_mb),
                str(self.overlay),
            ],
        )
        self.container = "sif-" + uuid.uuid4().hex
        if self.kind == "python":
            self._checked(self._launch("mkdir", "-p", "/work"))
        else:
            self._control(
                "exec", self.container, "bash", "-lc", "command -v timeout >/dev/null"
            )
            self._control(
                "exec", self.container, "git", "reset", "--hard", self.base_commit
            )
            self._control("exec", self.container, "git", "clean", "-ffd")
            if (
                self._control("exec", self.container, "git", "rev-parse", "HEAD")
                != self.base_commit
            ):
                raise RuntimeError(
                    "SIF repository does not match the frozen base commit"
                )

    def metadata(self) -> dict[str, Any]:
        return {
            **super().metadata(),
            "backend": "singularity",
            "sif": str(self.sif),
            "host_worker_capacity": self.host_worker_capacity,
            "resource_limits": "Requires site Slurm job cgroup; timeout/output bounded per call; no per-task CPU/RAM cgroup",
        }

    def close(self) -> None:
        self.container = None
        self.overlay.unlink(missing_ok=True)
        shutil.rmtree(self.scratch, ignore_errors=True)


class _PythonSIF(_SIFMixin, PythonSandbox):
    pass


class _RepositorySIF(_SIFMixin, RepositoryEnvironment):
    pass


class Host:
    """Private job-scoped registry; manifest images are pinned by actual SIF digest."""

    def __init__(
        self,
        root: Path,
        images: Path,
        *,
        executable="singularity",
        workers=4,
        overlay_mb=4096,
    ):
        if workers < 1 or overlay_mb < 64:
            raise ValueError("Positive workers and overlay size >=64 MiB are required")
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.root.chmod(0o700)
        self.executable, self.overlay_mb, self.workers = executable, overlay_mb, workers
        self.slots = threading.BoundedSemaphore(workers)
        self.lock = threading.RLock()
        self.tasks, self.aliases, self.images = {}, {}, {}
        for alias, entry in json.loads(images.read_text()).items():
            path = Path(entry["sif"]).resolve(strict=True)
            with path.open("rb") as stream:
                digest = (
                    hashlib.file_digest(stream, "sha256").hexdigest()
                    if hasattr(hashlib, "file_digest")
                    else _sha256(stream)
                )
            if path.suffix != ".sif" or digest != entry["sha256"]:
                raise ValueError("SIF digest does not match its image manifest")
            identity = "sha256:" + digest
            self.aliases[alias] = identity
            self.images[identity] = path

    def dispatch(self, action: str, **payload: Any) -> Any:
        if action == "resolve":
            image = payload["image"]
            identity = self.aliases.get(image, image)
            if identity not in self.images:
                raise ValueError("Image is not registered in this host's SIF manifest")
            return identity
        if action == "health":
            return {
                "backend": "singularity",
                "images": len(self.images),
                "pid": os.getpid(),
            }
        if action == "open":
            return self.open(**payload)
        if action == "grade":
            with self.slots:
                return self.grade(**payload)
        task_id = payload["task_id"]
        with self.lock:
            entry = self.tasks.get(task_id)
            if entry is None and action == "close":
                return {}
            runtime, lock = entry
        with lock:
            if action == "invoke":
                expected = (
                    "math_python" if runtime.kind == "python" else "workspace_exec"
                )
                if payload["name"] != expected:
                    raise ValueError("Unknown environment tool")
                with self.slots:
                    value = runtime.invoke(payload["name"], payload["arguments"])
                return {
                    "value": value,
                    "timing": runtime.call_timings[-1],
                }
            if action == "metadata":
                return runtime.metadata()
            if action == "export":
                with self.slots:
                    return runtime.export_artifacts()
            if action == "close":
                metadata = runtime.metadata()
                runtime.close()
                with self.lock:
                    self.tasks.pop(task_id, None)
                return metadata
        raise ValueError("Unknown host action")

    def open(
        self,
        kind: str,
        task_id: str,
        image: str,
        settings: dict[str, Any],
        options: dict[str, Any],
    ) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", task_id):
            raise ValueError("Task ID must be a safe single directory name")
        provider, settings_type = {
            "python": (_PythonSIF, ToolSettings),
            "repository": (_RepositorySIF, RepositorySettings),
        }[kind]
        identity = self.dispatch("resolve", image=image)
        settings = settings_type(**{**settings, "image": identity, "root": self.root})
        runtime = provider(
            settings,
            task_id,
            image_id=identity,
            **options,
            sif=self.images[identity],
            executable=self.executable,
            overlay_mb=self.overlay_mb,
        )
        runtime.host_worker_capacity = self.workers
        with self.lock:
            if task_id in self.tasks:
                raise ValueError("Task already exists")
            self.tasks[task_id] = (runtime, threading.RLock())

    def grade(
        self, task_id, image, repo, base_commit, patch, test_patch, test_cmds, timeout
    ):
        if not 0 < timeout <= 3600:
            raise ValueError("Grading timeout must be within 3600 seconds")
        self.open(
            "repository",
            task_id,
            image,
            {"timeout": timeout, "max_output_bytes": 64 * 1024 * 1024},
            {"repo": repo, "base_commit": base_commit},
        )
        runtime, lock = self.tasks[task_id]
        try:
            with lock:
                runtime.patches = runtime.directory / "patches"
                runtime.patches.mkdir()
                (runtime.patches / "patch.diff").write_text(patch)
                (runtime.patches / "test.diff").write_text(test_patch)
                result = json.loads(
                    runtime.execute(
                        "\n".join(
                            [
                                "set -e",
                                "git apply -v --3way --recount --ignore-space-change --whitespace=nowarn /patches/patch.diff",
                                "git apply -v --3way --recount --ignore-space-change --whitespace=nowarn /patches/test.diff",
                                *test_cmds,
                            ]
                        )
                    )
                )
                if result["status"] != "completed" or result["exit_code"] in (124, 125):
                    raise RuntimeError(
                        "SIF test execution could not be scored completely"
                    )
                return [result["exit_code"], result["output"]]
        finally:
            self.dispatch("close", task_id=task_id)

    def close(self):
        for task_id in list(self.tasks):
            self.dispatch("close", task_id=task_id)


def _sha256(stream: Any) -> str:
    digest = hashlib.sha256()
    while chunk := stream.read(1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest()


class _Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        try:
            result = {"result": self.server.host.dispatch(**_read(self.rfile))}
        except Exception as exc:
            result = {"error": f"{type(exc).__name__}: {exc}"}
        self.wfile.write(json.dumps(result).encode() + b"\n")


class HostServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True

    def __init__(self, path: Path, host: Host) -> None:
        self.host = host
        super().__init__(str(path), _Handler)
        Path(path).chmod(0o600)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve")
    serve.add_argument("--socket", type=Path, required=True)
    serve.add_argument("--root", type=Path, required=True)
    serve.add_argument("--images", type=Path, required=True)
    serve.add_argument("--executable", default="singularity")
    serve.add_argument("--workers", type=int, default=4)
    serve.add_argument("--overlay-mb", type=int, default=4096)
    commands.add_parser("health")
    args = parser.parse_args()
    if args.command == "health":
        print(json.dumps(rpc("health")))
        return
    os.umask(0o077)
    args.socket.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    host = Host(
        args.root,
        args.images,
        executable=args.executable,
        workers=args.workers,
        overlay_mb=args.overlay_mb,
    )

    def stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    bound = False
    try:
        with HostServer(args.socket, host) as server:
            bound = True
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
    finally:
        host.close()
        if bound:
            args.socket.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
