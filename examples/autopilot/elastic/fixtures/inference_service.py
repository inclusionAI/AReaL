# SPDX-License-Identifier: Apache-2.0

"""SWE/validation fixture owning SGLang and attaching AReaL v2 CPU services."""

import fcntl
import os
import socket
import subprocess
import threading
import time
import uuid
from pathlib import Path

import httpx
import psutil

from areal.api.cli_args import InferenceEngineConfig, SchedulingSpec, SGLangConfig
from areal.api.io_struct import LocalInfServerInfo
from areal.autopilot.elastic.runtime.processes import (
    OWNER_ENV,
    gpu_processes,
    resolve_gpu_uuids,
    stop_owned_processes,
)
from areal.infra.scheduler.local import LocalScheduler
from areal.v2.inference_service.controller.controller import RolloutControllerV2


class InferenceService:
    """Local SGLang replica/TP owner on an otherwise idle, cooperative GPU pool."""

    def __init__(
        self,
        *,
        model: Path,
        model_name: str,
        root: Path,
        lock_root: Path,
        device: int | None = None,
        devices: list[int] | None = None,
        rollout_config=None,
        context_length: int = 16384,
    ):
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise RuntimeError("Inference coordinator must have CUDA disabled")
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.devices = list(devices) if devices is not None else [device]
        if (
            not self.devices
            or any(type(d) is not int for d in self.devices)
            or len(set(self.devices)) != len(self.devices)
        ):
            raise ValueError("Inference GPU IDs must be nonempty and unique")
        self.layout, self.context_length = rollout_config, context_length
        if (
            rollout_config is not None
            and len(self.devices) != rollout_config.world_size
        ):
            raise ValueError("Inference devices do not match replica/TP configuration")
        self.device = self.devices[0]
        self.gpu_uuids = resolve_gpu_uuids(self.devices)
        self.gpu_uuid = self.gpu_uuids[0]
        lock_root.mkdir(parents=True, exist_ok=True)
        self._locks = []
        try:
            for gpu in sorted(self.gpu_uuids):
                lock = (lock_root / f"{gpu}.lock").open("a")
                self._locks.append(lock)
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            for lock in self._locks:
                lock.close()
            raise
        self._lock = self._locks[0]
        self.model, self.model_name = model, model_name
        self.backend = None
        self.backends, self.logs, self.backend_settings = [], [], []
        self.controller = None
        self.log = None
        self._owned: set[tuple[int, float]] = set()
        self._lifecycle_lock = threading.Lock()
        self.owner = uuid.uuid4().hex
        self._closed = False

    def start(self, timeout: float = 300) -> str:
        # asyncio cancellation does not cancel to_thread: close must serialize
        # with the entire startup, including the period before controller exists.
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("Inference service has been closed")
            return self._start(timeout)

    def _start(self, timeout: float) -> str:
        if self.backends:
            raise RuntimeError("Inference service already started")
        occupied = gpu_processes()
        if any(occupied.get(gpu) for gpu in self.gpu_uuids):
            raise RuntimeError("Inference GPU is occupied")
        replicas = 1 if self.layout is None else self.layout.replicas
        tp = 1 if self.layout is None else self.layout.tp
        infos = [self._start_backend(index, tp, timeout) for index in range(replicas)]
        return self._start_controller(infos, replicas, tp)

    def _start_backend(self, replica, tp, timeout):
        with socket.socket() as address:
            address.bind(("127.0.0.1", 0))
            port = address.getsockname()[1]
        config = SGLangConfig(
            model_path=str(self.model),
            mem_fraction_static=0.2,
            max_running_requests=1
            if self.layout is None
            else self.layout.max_batch_sequences,
            context_length=self.context_length,
            max_prefill_tokens=self.context_length
            if self.layout is None
            else self.layout.max_batch_tokens,
            disable_cuda_graph=True,
            enable_metrics=False,
        )
        args = SGLangConfig.build_args(
            config, tp_size=tp, base_gpu_id=0, host="127.0.0.1", port=port
        )
        if self.layout is not None:
            args["max_total_tokens"] = self.layout.kv_capacity_tokens
        log = (
            self.root / ("sglang.log" if replica == 0 else f"sglang-{replica}.log")
        ).open("w")
        self.logs.append(log)
        backend = subprocess.Popen(
            SGLangConfig.build_cmd_from_args(args),
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": ",".join(
                    self.gpu_uuids[replica * tp : (replica + 1) * tp]
                ),
                OWNER_ENV: self.owner,
                "TRITON_CACHE_DIR": os.environ.get(
                    "TRITON_CACHE_DIR", str(self.root / "triton-cache")
                ),
                "TORCHINDUCTOR_CACHE_DIR": os.environ.get(
                    "TORCHINDUCTOR_CACHE_DIR", str(self.root / "inductor-cache")
                ),
                "TVM_FFI_CACHE_DIR": os.environ.get(
                    "TVM_FFI_CACHE_DIR", str(self.root / "tvm-ffi-cache")
                ),
            },
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.backends.append(backend)
        self.backend, self.log = self.backends[0], self.logs[0]
        backend_url = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + timeout
        with httpx.Client(timeout=3, trust_env=False) as client:
            while True:
                if backend.poll() is not None:
                    raise RuntimeError(
                        f"SGLang replica {replica} exited with {backend.returncode}"
                    )
                try:
                    if client.get(f"{backend_url}/health").status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError("SGLang did not become healthy")
                time.sleep(1)
            if self.layout is not None:
                response = client.get(f"{backend_url}/get_server_info")
                response.raise_for_status()
                info = response.json()
                reported = info.get("server_args", info)
                fields = (
                    "tp_size",
                    "max_running_requests",
                    "context_length",
                    "max_prefill_tokens",
                    "max_total_tokens",
                )
                for name in fields:
                    if reported.get(name) != args[name]:
                        raise RuntimeError(
                            f"SGLang did not apply {name}: {reported.get(name)!r}"
                        )
                self.backend_settings.append({name: reported[name] for name in fields})
        return LocalInfServerInfo(host="127.0.0.1", port=port, process=backend)

    def _start_controller(self, infos, replicas, tp):
        if len(infos) != replicas:
            raise RuntimeError("One backend is required per Rollout replica")
        run_id = uuid.uuid4().hex
        (self.root / "names").mkdir(exist_ok=True)
        scheduler = LocalScheduler(
            gpu_devices=list(self.devices),
            log_dir=str(self.root / "workers"),
            startup_timeout=120,
            experiment_name="elastic-swe",
            trial_name=run_id,
            fileroot=str(self.root),
            nfs_record_root=str(self.root / "names"),
        )
        self.admin_key = uuid.uuid4().hex
        config = InferenceEngineConfig(
            backend=f"sglang:d{replicas}t{tp}",
            tokenizer_path=str(self.model),
            model=self.model_name,
            experiment_name="elastic-swe",
            trial_name=run_id,
            fileroot=str(self.root),
            scheduling_spec=(
                SchedulingSpec(
                    gpu=0,
                    cpu=1,
                    mem=4,
                    env_vars={"CUDA_VISIBLE_DEVICES": "", OWNER_ENV: self.owner},
                ),
            ),
            max_concurrent_rollouts=1,
            consumer_batch_size=1,
            max_head_offpolicyness=0,
            setup_timeout=180,
            workers_ready_timeout=120,
            admin_api_key=self.admin_key,
            _version="v2",
        )
        self.controller = RolloutControllerV2(config, scheduler)
        self.controller.initialize(role="rollout", server_infos=infos, wait=True)
        return self.controller._gateway_addr

    def assert_drained(self):
        """All clients must hold the episode gate through export and session removal."""
        stats = self.controller.staleness_manager.get_stats()
        if stats.running or stats.enqueued:
            raise ValueError("Rollout workflows have not drained")
        with httpx.Client(timeout=10, trust_env=False) as client:
            for proxy in self.controller._data_proxy_addrs:
                response = client.get(f"{proxy}/health")
                response.raise_for_status()
                if response.json()["sessions"] != 0:
                    raise ValueError(
                        "Finalize, export and remove Rollout sessions before reconfiguration"
                    )

    def verify_version(self, version):
        if (
            not self.backends
            or any(p.poll() is not None for p in self.backends)
            or self.controller.get_version() != version
        ):
            return False
        with httpx.Client(timeout=10, trust_env=False) as client:
            for proxy in self.controller._data_proxy_addrs:
                response = client.get(f"{proxy}/get_version")
                response.raise_for_status()
                if response.json().get("version") != version:
                    return False
        return True

    def resume(self):
        self.controller.continue_generation()
        self.controller.resume()
        # Core fanout helpers can log partial failures without raising.
        with httpx.Client(timeout=10, trust_env=False) as client:
            for backend in self.controller._inf_addrs:
                client.get(f"{backend}/health").raise_for_status()
            for proxy in self.controller._data_proxy_addrs:
                response = client.get(f"{proxy}/health")
                response.raise_for_status()
                if response.json().get("paused") is not False:
                    raise RuntimeError("A Rollout replica is still paused")

    def close(self) -> dict:
        with self._lifecycle_lock:
            return self._close()

    def _close(self) -> dict:
        if self.controller is not None:
            future = self.controller._init_future
            self.controller._shutdown_requested.set()
            if future is not None:
                try:
                    future.result(timeout=240)
                except Exception:
                    if not future.done():
                        raise RuntimeError(
                            "Inference initialization is still running; retain ownership lock"
                        ) from None
        # Snapshot only the backend and guards belonging to this service.
        roots = list(self.backends)
        if self.controller is not None:
            roots.extend(
                info.process
                for role in self.controller._service_roles
                for info in self.controller.scheduler._workers.get(role, [])
                if info.process is not None
            )
        for parent in roots:
            try:
                process = psutil.Process(parent.pid)
                self._owned.update(
                    (child.pid, child.create_time())
                    for child in [process, *process.children(recursive=True)]
                )
            except psutil.NoSuchProcess:
                pass
        try:
            if self.controller is not None:
                self.controller.destroy()
        finally:
            try:
                stop_owned_processes(self.owner)
                for backend in self.backends:
                    backend.wait(timeout=10)
            finally:
                for log in self.logs:
                    log.close()
        deadline = time.monotonic() + 15
        while True:
            alive = []
            for pid, created in self._owned:
                try:
                    process = psutil.Process(pid)
                    if (
                        process.create_time() == created
                        and process.status() != psutil.STATUS_ZOMBIE
                    ):
                        alive.append(pid)
                except psutil.NoSuchProcess:
                    pass
            occupied = gpu_processes()
            gpu_pids = {
                gpu: occupied[gpu] for gpu in self.gpu_uuids if occupied.get(gpu)
            }
            if not alive and not gpu_pids:
                for lock in self._locks:
                    lock.close()
                self._closed = True
                return {
                    "gpu_uuid": self.gpu_uuid,
                    "gpu_uuids": list(self.gpu_uuids),
                    "owned_processes_exited": True,
                    "gpu_free": True,
                }
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"Inference cleanup unverified: live={alive}, GPU={gpu_pids}"
                )
            time.sleep(0.5)
