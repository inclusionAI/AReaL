# SPDX-License-Identifier: Apache-2.0

"""Persistent CPU training services and the HTTP endpoint used while suspended."""

import argparse
import hashlib
import secrets
import sys
import time
from dataclasses import asdict

import psutil
import requests
from fastapi import FastAPI
from fastapi.responses import JSONResponse


class TrainingFrontend:
    role = "actor-frontend-guard"

    def __init__(self, engine_class, config, scheduler):
        from areal.v2.training_service.controller.controller import (
            GatewayTrainController,
        )

        self.config = config
        self.scheduler = scheduler
        # Reuse the upstream guard/fork/cleanup helpers without creating a PG.
        self.control = GatewayTrainController(engine_class, config, scheduler)
        self.control._role = "actor-frontend"
        self.api_key = secrets.token_urlsafe(32)
        self.endpoints: dict[str, tuple[str, int]] = {}
        self.active_generation: str | None = None
        self._baseline: list[dict] | None = None
        self._closing_processes: set[tuple[int, float]] = set()
        self.closed = False

    def address(self, role: str) -> str:
        from areal.utils.network import format_hostport

        return f"http://{format_hostport(*self.endpoints[role])}"

    def start(self) -> dict:
        from areal.api.cli_args import SchedulingSpec
        from areal.api.scheduler_api import Job
        from areal.infra.utils.concurrent import run_async_task
        from areal.utils.network import format_hostport

        if self.closed:
            raise RuntimeError("Closed frontend cannot be restarted")
        if self.endpoints:
            return self.probe(suspended=True)
        started = time.monotonic()
        spec = SchedulingSpec(**asdict(self.config.scheduling_spec[0]))
        spec.cpu, spec.gpu = 2, 0
        spec.cmd = "python -m areal.v2.training_service.guard"
        spec.env_vars = {**spec.env_vars, "CUDA_VISIBLE_DEVICES": ""}
        self.control._service_roles.append(self.role)
        try:
            self.scheduler.create_workers(Job(role=self.role, replicas=1, tasks=[spec]))
            worker = self.scheduler.get_workers(
                self.role, timeout=int(self.config.setup_timeout)
            )[0]
            guard = f"http://{format_hostport(worker.ip, int(worker.worker_ports[0]))}"

            async def fork_services():
                for role in ("router", "standby", "gateway"):
                    module = (
                        "areal.autopilot.elastic.runtime.training.service"
                        if role == "standby"
                        else f"areal.v2.training_service.{role}"
                    )
                    cmd = [sys.executable, "-m", module]
                    if role != "standby":
                        cmd += ["--admin-api-key", self.config.admin_api_key]
                    if role == "gateway":
                        cmd += [
                            "--router-addr",
                            self.address("router"),
                            "--forward-timeout",
                            str(self.config.request_timeout),
                        ]
                    self.endpoints[role] = await self.control._async_fork_on_guard(
                        guard, role, 0, cmd, {"CUDA_VISIBLE_DEVICES": ""}
                    )

            run_async_task(fork_services)
            self._register(self.address("standby"))
            result = self.probe(suspended=True)
            return {**result, "startup_seconds": time.monotonic() - started}
        except BaseException:
            self.close()
            raise

    def _register(self, model_addr: str) -> None:
        from areal.infra.utils.concurrent import run_async_task

        run_async_task(
            self.control._register_in_router,
            self.address("router"),
            model_addr,
            self.api_key,
        )

    def publish(self, actor, step: int) -> dict:
        """Called only after private staging-key initialization and recovery succeed."""
        if self.active_generation is not None:
            raise RuntimeError("A training generation is already published")
        # Claim ownership before HTTP: a lost response may still have committed.
        self.active_generation = actor.launch_id
        self._register(actor._model_addr)
        actor.api_key = self.api_key
        if actor._gateway_get_result("/get_version") != step:
            raise RuntimeError("Published worker version differs from restored step")
        return self.probe(suspended=False)

    def park(self, generation: str) -> bool:
        """A retired generation must never withdraw its successor's route."""
        if self.active_generation != generation:
            return False
        self._register(self.address("standby"))
        self.active_generation = None
        self.probe(suspended=True)
        return True

    def _process_snapshot(self) -> list[dict]:
        info = self.scheduler._workers[self.role][0]
        if info.gpu_devices:
            raise RuntimeError("Frontend guard unexpectedly owns GPU resources")
        parent = psutil.Process(info.process.pid)
        records = []
        # The upstream log streamer inserts shell/tee processes around services.
        for process in [parent, *parent.children(recursive=True)]:
            cmd = process.cmdline()
            if "-m" not in cmd:
                continue
            module = cmd[cmd.index("-m") + 1]
            if module not in {
                "areal.v2.training_service.guard",
                "areal.v2.training_service.router",
                "areal.v2.training_service.gateway",
                "areal.autopilot.elastic.runtime.training.service",
            }:
                continue
            visible = process.environ().get("CUDA_VISIBLE_DEVICES")
            if visible != "" or process.status() == psutil.STATUS_ZOMBIE:
                raise RuntimeError("Frontend process must be alive and CPU-only")
            records.append(
                {
                    "module": module,
                    "pid": process.pid,
                    "created": process.create_time(),
                    "cuda_visible_devices": visible,
                }
            )
        records.sort(key=lambda item: item["module"])
        if len(records) != 4:
            raise RuntimeError(
                f"Expected four persistent CPU processes, got {len(records)}"
            )
        if self._baseline is None:
            self._baseline = records
        elif records != self._baseline:
            raise RuntimeError("Persistent frontend process identity changed")
        return records

    def probe(self, *, suspended: bool) -> dict:
        for role in ("gateway", "router", "standby"):
            requests.get(f"{self.address(role)}/health", timeout=10).raise_for_status()
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = requests.get(
            f"{self.address('gateway')}/get_version", headers=headers, timeout=10
        )
        expected = 503 if suspended else 200
        if response.status_code != expected:
            raise RuntimeError(
                f"Stable training endpoint returned {response.status_code}, expected {expected}"
            )
        if suspended:
            response = requests.post(
                f"{self.address('gateway')}/sft/train",
                headers=headers,
                json={},
                timeout=10,
            )
            if (
                response.status_code != 503
                or response.json().get("error") != "training_unavailable"
            ):
                raise RuntimeError(
                    "Suspended endpoint did not reject training explicitly"
                )
        return {
            "services": {role: self.address(role) for role in self.endpoints},
            "processes": self._process_snapshot(),
            "api_key_sha256": hashlib.sha256(self.api_key.encode()).hexdigest(),
            "training_status": expected,
        }

    def close(self) -> dict:
        if self.closed:
            return {"process_exit_verified": True}
        started = time.monotonic()
        for info in self.scheduler._workers.get(self.role, []):
            if info.process is None:
                continue
            try:
                parent = psutil.Process(info.process.pid)
                self._closing_processes.update(
                    (p.pid, p.create_time())
                    for p in [parent, *parent.children(recursive=True)]
                )
            except psutil.NoSuchProcess:
                pass
        self.control.destroy()
        deadline = time.monotonic() + 30
        while True:
            alive = []
            for pid, created in self._closing_processes:
                try:
                    p = psutil.Process(pid)
                    if (
                        p.create_time() == created
                        and p.status() != psutil.STATUS_ZOMBIE
                    ):
                        alive.append(pid)
                except psutil.NoSuchProcess:
                    pass
            if not alive:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Frontend processes failed to exit: {alive}")
            time.sleep(0.2)
        self.closed = True
        return {
            "process_exit_verified": True,
            "shutdown_seconds": time.monotonic() - started,
        }


def create_app() -> FastAPI:
    app = FastAPI(title="Elastic training standby")

    @app.get("/health")
    async def health():
        return {"status": "ok", "training_available": False}

    @app.api_route("/{path:path}", methods=["GET", "POST"])
    async def unavailable(path: str):
        return JSONResponse(
            {"error": "training_unavailable", "retryable": True},
            status_code=503,
            headers={"Retry-After": "1"},
        )

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    args = parser.parse_args()

    import uvicorn

    from areal.infra.utils.http import get_default_uvicorn_kwargs

    uvicorn.run(
        create_app(),
        host=args.host,
        port=args.port,
        access_log=False,
        log_level="warning",
        **get_default_uvicorn_kwargs(),
    )


if __name__ == "__main__":
    main()
