# SPDX-License-Identifier: Apache-2.0

"""Start, restore and release training workers, including the local v2 adapter."""

import asyncio
import dataclasses
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Any

import psutil

from areal.api import SaveLoadMeta
from areal.autopilot.contracts import TrainingConfig
from areal.autopilot.elastic.runtime.processes import gpu_processes, resolve_gpu_uuids
from areal.autopilot.elastic.runtime.state.checkpoint import (
    CheckpointStore,
    atomic_json,
    recovery_digest,
    recovery_view,
)
from areal.infra import TrainController
from areal.infra.scheduler.local import LocalScheduler
from areal.infra.utils.concurrent import run_async_task


class TrainGroup:
    def __init__(
        self,
        *,
        devices: list[int],
        config: Any,
        engine_class: type | str,
        ft_spec: Any,
        root: Path,
        compatibility: dict[str, Any],
        keep_frontend: bool = False,
    ):
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise RuntimeError(
                "Run the CPU coordinator with CUDA_VISIBLE_DEVICES empty"
            )
        if not devices or len(set(devices)) != len(devices):
            raise ValueError("Training GPU IDs must be nonempty and unique")
        self.pool_devices = list(devices)
        self.pool_uuids = resolve_gpu_uuids(devices)
        self.devices = list(devices)
        self.uuids = list(self.pool_uuids)
        self._world_size = len(devices)
        self._placement = None
        self._recovery_view = None
        self.config = config
        self.is_v2 = config._version == "v2"
        if keep_frontend and not self.is_v2:
            raise ValueError("Persistent frontend requires the v2 controller")
        self.worker_role = "actor-guard" if self.is_v2 else "actor"
        self.engine_class = engine_class
        self.ft_spec = ft_spec
        self.compatibility = compatibility
        self.root = root
        self.store = CheckpointStore(root / "checkpoints")
        (root / "name-resolve").mkdir(parents=True, exist_ok=True)
        self.scheduler = LocalScheduler(
            gpu_devices=list(devices),
            log_dir=str(root / "workers"),
            startup_timeout=120,
            experiment_name=config.experiment_name,
            trial_name=config.trial_name,
            fileroot=str(root),
            nfs_record_root=str(root / "name-resolve"),
        )
        self.actor = None
        self.frontend = None
        if keep_frontend:
            from areal.autopilot.elastic.runtime.training.service import (
                TrainingFrontend,
            )

            self.frontend = TrainingFrontend(engine_class, config, self.scheduler)
        self.step = 0
        self.events: list[dict[str, Any]] = []
        self._started_processes: set[tuple[int, float]] = set()
        self._release_pending = False
        self._gpu_release_verified = False
        self._releasing_processes: set[tuple[int, float]] = set()
        self._releasing_launch_id: str | None = None

    @property
    def training_config(self):
        return TrainingConfig(self._world_size)

    def validate_config(self, target: TrainingConfig) -> None:
        """Side-effect-free capability check, including before worker suspension."""
        if target.profile_id:
            raise ValueError(
                "TrainGroup cannot execute named actor profiles; only pure-DP resizing is supported"
            )
        if target.world_size > len(self.pool_devices):
            raise ValueError("Configuration exceeds training GPU pool")
        if target.world_size == getattr(self, "_world_size", None):
            return
        if self.compatibility.get("elastic_dp") != "fsdp-pure-dp-v1" or not self.is_v2:
            raise ValueError("Group does not support data-parallel resizing")
        specs = self.config.scheduling_spec
        if (
            len(specs) != 1
            or specs[0].gpu != 1
            or self.config.scheduling_strategy.type != "separation"
        ):
            raise ValueError("Resizable group requires one GPU per separated rank")

    def configure(self, target: TrainingConfig):
        """Change only the pure-DP layout of a fully suspended generation."""
        self.validate_config(target)
        if self.actor is not None or self._release_pending:
            raise RuntimeError("Cannot configure before verified GPU release")
        if target.world_size == getattr(self, "_world_size", None):
            return
        self.config = dataclasses.replace(
            self.config, backend=f"fsdp:d{target.world_size}"
        )
        self._world_size = target.world_size

    def _select_devices(self):
        if getattr(self, "pool_devices", None) is None:
            return
        if getattr(self, "_placement", None) is not None:
            if len(self._placement) != self._world_size:
                raise ValueError("Pinned Actor placement differs from configuration")
            self.devices = list(self._placement)
            self.uuids = [
                self.pool_uuids[self.pool_devices.index(d)] for d in self.devices
            ]
            return
        offset = self.scheduler._gpu_counter
        indices = [
            (offset + rank) % len(self.pool_devices) for rank in range(self._world_size)
        ]
        self.devices = [self.pool_devices[index] for index in indices]
        self.uuids = [self.pool_uuids[index] for index in indices]

    def configure_placement(self, devices: list[int]) -> None:
        """Pin a suspended Actor to the joint allocator's nonoverlapping devices."""
        if self.actor is not None or self._release_pending:
            raise RuntimeError("Release Actor before changing placement")
        if (
            len(devices) != self._world_size
            or len(set(devices)) != len(devices)
            or not set(devices).issubset(self.pool_devices)
        ):
            raise ValueError("Invalid Actor placement within the shared GPU pool")
        self._placement = tuple(devices)

    def _event(self, state: str, **extra):
        self.events.append({"state": state, "global_step": self.step, **extra})
        atomic_json(self.root / "lifecycle.json", self.events)

    def _require_free_gpus(self):
        processes = gpu_processes()
        occupied = {uuid: processes[uuid] for uuid in self.uuids if processes.get(uuid)}
        if occupied:
            raise RuntimeError(f"Training GPU pool is occupied: {occupied}")

    def start(self, *, restore: bool = False):
        if self._release_pending:
            raise RuntimeError("Previous release is unverified; retry stop first")
        if self.actor is not None:
            raise RuntimeError("Training group already exists")
        self._select_devices()
        self._require_free_gpus()
        self._gpu_release_verified = False
        manifest = None
        checkpoint = None
        if restore:
            checkpoint, manifest = self.store.latest(self.compatibility)
            if (
                manifest["world_size"] != len(self.devices)
                and self.compatibility.get("elastic_dp") != "fsdp-pure-dp-v1"
            ):
                raise ValueError("Checkpoint does not support resizing")
            if self.is_v2:
                checkpoint = recovery_view(
                    checkpoint,
                    manifest,
                    target_world_size=len(self.devices),
                    root=self.root / "recovery",
                )
                self._recovery_view = checkpoint
        started = time.monotonic()
        self._event("STARTING", restoring=restore)
        if self.frontend is not None:
            first_start = not self.frontend.endpoints
            frontend_state = self.frontend.start()
            if first_start:
                self._event("FRONTEND_READY", frontend=frontend_state)
        if self.is_v2:
            from areal.autopilot.elastic.runtime.training.workers import (
                LocalV2TrainController,
            )

            self.actor = LocalV2TrainController(
                self.engine_class,
                self.config,
                self.scheduler,
                gpu_uuids=self.uuids,
                frontend=self.frontend,
            )
        else:
            self.actor = TrainController(self.engine_class, self.config, self.scheduler)
        try:
            self.actor.create_process_group(self.actor.train_alloc.parallel)
            self.actor.initialize(
                role="actor",
                addr=None,
                ft_spec=self.ft_spec,
                **({"wait": True} if self.is_v2 else {"seed": 17}),
            )
            if self.frontend is not None:
                self._event(
                    "STAGING",
                    phase="initialized",
                    frontend=self.frontend.probe(suspended=True),
                )
            if restore:
                self.actor.load(
                    SaveLoadMeta(
                        path=str(checkpoint / "dcp"),
                        weight_format="dcp",
                        with_optim=True,
                    )
                )
                states = [
                    json.loads((checkpoint / f"runtime-{rank}.json").read_text())
                    for rank in range(len(self.devices))
                ]
                if self.is_v2:
                    restored = [
                        record["state"]
                        for record in self.actor.rank_records("restored")
                    ]
                else:
                    restored = self.call_all(
                        "restore_runtime", per_rank_args=[(state,) for state in states]
                    )
                # JSON normalizes tuples in the Python RNG state.
                if manifest["world_size"] == len(self.devices):
                    if json.loads(json.dumps(restored)) != states:
                        raise RuntimeError(
                            "Restored runtime state differs from checkpoint"
                        )
                else:
                    records = self.actor.rank_records("restored")
                    for rank, (actual, plan, record) in enumerate(
                        zip(restored, states, records, strict=True)
                    ):
                        if (
                            (
                                actual["rank"],
                                actual["world_size"],
                                actual["policy_version"],
                            )
                            != (rank, len(self.devices), manifest["global_step"])
                            or actual["lr_scheduler"] != plan["lr_scheduler"]
                            or record.get("recovery_sha256") != recovery_digest(plan)
                        ):
                            raise RuntimeError(
                                "Resized runtime differs from recovery plan"
                            )
                self.step = manifest["global_step"]
                # v2 also caches the version in the CPU controller.
                self.actor.set_version(self.step)
                if self.frontend is not None:
                    self._event(
                        "STAGING",
                        phase="restored",
                        frontend=self.frontend.probe(suspended=True),
                    )
            if self.is_v2:
                self.actor.train()
            else:
                self.call_all("set_training_mode")
            frontend_state = None
            if self.frontend is not None:
                if self.actor._gateway_get_result("/get_version") != self.step:
                    raise RuntimeError(
                        "Staged worker version differs from restored step"
                    )
                frontend_state = self.frontend.publish(self.actor, self.step)
            identities = [
                (info.process.pid, psutil.Process(info.process.pid).create_time())
                for info in self.scheduler._workers[self.worker_role]
            ]
            if self._started_processes.intersection(identities):
                raise RuntimeError("Training restart reused an existing worker process")
            self._started_processes.update(identities)
            self._event(
                "ACTIVE",
                startup_seconds=time.monotonic() - started,
                world_size=len(self.devices),
                devices=list(self.devices),
                recovery_mode=(
                    "exact"
                    if manifest is not None
                    and manifest["world_size"] == len(self.devices)
                    else "reseed"
                )
                if restore
                else "initial",
                worker_processes=identities,
                **({"frontend": frontend_state} if frontend_state is not None else {}),
                **(
                    {
                        "services": self.actor.service_endpoints(),
                        "train_workers": self.actor.rank_records("process"),
                    }
                    if self.is_v2
                    else {}
                ),
            )
        except BaseException:
            self.stop()
            self._clear_recovery_view()
            raise
        self._clear_recovery_view()

    def _clear_recovery_view(self):
        view = getattr(self, "_recovery_view", None)
        if view is not None:
            shutil.rmtree(view)
            self._recovery_view = None

    def call_all(self, method: str, *args, per_rank_args=None):
        if self.actor is None:
            raise RuntimeError("Training group is not active")

        async def call():
            results = await asyncio.gather(
                *[
                    self.scheduler.async_call_engine(
                        worker.id,
                        method,
                        f"actor/{rank}",
                        *(per_rank_args[rank] if per_rank_args is not None else args),
                    )
                    for rank, worker in enumerate(self.actor.workers)
                ],
                return_exceptions=True,
            )
            for result in results:
                if isinstance(result, BaseException):
                    raise result
            return results

        return run_async_task(call)

    def checkpoint(self, *, data_cursor: dict[str, Any] | None = None):
        if self.actor is None:
            raise RuntimeError("Cannot checkpoint a suspended training group")
        self._event("SAVING")
        started = time.monotonic()

        def write(path):
            states = None if self.is_v2 else self.call_all("capture_runtime")
            self.actor.save(
                SaveLoadMeta(
                    path=str(path / "dcp"),
                    weight_format="dcp",
                    with_optim=True,
                )
            )
            if states is not None:
                for rank, state in enumerate(states):
                    atomic_json(path / f"runtime-{rank}.json", state)
            self._checkpoint_extra(path)

        path = self.store.commit(
            step=self.step,
            world_size=len(self.devices),
            compatibility=self.compatibility,
            write=write,
            estimated_bytes=10 << 30,
            data_cursor=data_cursor,
            data_progress=getattr(self, "data_progress", None),
        )
        size = sum(file.stat().st_size for file in path.rglob("*") if file.is_file())
        self._event(
            "ACTIVE",
            checkpoint_bytes=size,
            checkpoint_seconds=time.monotonic() - started,
        )
        return path

    def _checkpoint_extra(self, path: Path) -> None:
        """Write optional serving artifacts before checkpoint publication."""

    def stop(self):
        if self.actor is None and not getattr(self, "_release_pending", False):
            self._clear_recovery_view()
            return
        event_errors = []

        def record(state, **extra):
            # A full or failed checkpoint disk must never prevent GPU cleanup.
            try:
                self._event(state, **extra)
            except OSError as error:
                event_errors.append(error)

        record("STOPPING")
        started = time.monotonic()
        if self.actor is not None and self.is_v2:
            try:
                self.actor.quiesce_initialization()
            except RuntimeError as error:
                record("FAILED", release_unverified=True, error=str(error))
                raise
        if self.actor is not None and self.frontend is not None:
            self._releasing_launch_id = self.actor.launch_id
        frontend_error = None
        if self.frontend is not None:
            try:
                self.frontend.park(self._releasing_launch_id)
                self.frontend.probe(suspended=True)
            except Exception as error:
                # Failed routing must not bypass GPU teardown. Do not report
                # successful suspension even if the following release succeeds.
                frontend_error = error
        identities = list(getattr(self, "_releasing_processes", set()))
        if self.actor is not None and self.is_v2:
            identities.extend(self.actor.deferred_processes)
        # Snapshot before destroy: scheduler removes its process records afterward.
        for info in self.scheduler._workers.get(self.worker_role, []):
            if info.process is None:
                continue
            try:
                parent = psutil.Process(info.process.pid)
                identities.extend(
                    (p.pid, p.create_time())
                    for p in [parent, *parent.children(recursive=True)]
                )
            except psutil.NoSuchProcess:
                pass
        self._releasing_processes = set(identities)
        self._release_pending = True
        if self.actor is not None:
            self._gpu_release_verified = False
            self.actor.destroy()
            self.actor = None
        deadline = time.monotonic() + 60
        while True:
            alive = []
            for pid, created in identities:
                try:
                    process = psutil.Process(pid)
                    if (
                        process.create_time() == created
                        and process.status() != psutil.STATUS_ZOMBIE
                    ):
                        alive.append(pid)
                except psutil.NoSuchProcess:
                    pass
            processes = gpu_processes()
            occupied = {
                uuid: processes[uuid] for uuid in self.uuids if processes.get(uuid)
            }
            if not alive and not occupied:
                break
            if time.monotonic() >= deadline:
                record("FAILED", surviving_pids=alive, gpu_processes=occupied)
                raise RuntimeError(
                    f"Training resources not released: {alive}, {occupied}"
                )
            time.sleep(1)
        # A separate process must be able to use all four cards after release.
        probe = (
            "import torch; xs=[torch.ones(16,device=f'cuda:{i}') for i in range(torch.cuda.device_count())]; assert len(xs)=="
            + str(len(self.devices))
            + "; [torch.cuda.synchronize(i) for i in range(len(xs))]"
        )
        subprocess.run(
            [sys.executable, "-c", probe],
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": ",".join(self.uuids),
            },
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self._gpu_release_verified = True
        if frontend_error is not None:
            record("FAILED", frontend_error=str(frontend_error))
            raise RuntimeError(
                "GPUs released but frontend suspension failed"
            ) from frontend_error
        frontend_state = None
        if self.frontend is not None:
            try:
                frontend_state = self.frontend.probe(suspended=True)
            except Exception as error:
                record("FAILED", frontend_error=str(error))
                raise
        record(
            "SUSPENDED",
            release_seconds=time.monotonic() - started,
            process_exit_verified=True,
            gpu_reallocation_verified=True,
            **({"frontend": frontend_state} if frontend_state is not None else {}),
        )
        self._release_pending = False
        self._releasing_processes.clear()
        self._releasing_launch_id = None
        if event_errors:
            raise RuntimeError(
                "Training resources released, but lifecycle events could not be persisted"
            ) from event_errors[0]

    def close(self):
        """Final experiment shutdown; ordinary stop() preserves CPU services."""
        try:
            self.stop()
        finally:
            # A failed event write need not leak CPU services, but unverified
            # GPU teardown must retain the frontend for a later cleanup retry.
            if (
                self.frontend is not None
                and self.actor is None
                and (
                    not self._release_pending
                    or getattr(self, "_gpu_release_verified", False)
                )
            ):
                result = self.frontend.close()
                self._event("FRONTEND_CLOSED", **result)


def __getattr__(name: str):
    """Load the v2 adapter only when requested, keeping v1 imports independent."""
    if name != "LocalV2TrainController":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from areal.v2.training_service.controller.controller import GatewayTrainController

    class LocalV2TrainController(GatewayTrainController):
        def __init__(self, *args, gpu_uuids: list[str], frontend=None, **kwargs):
            super().__init__(*args, **kwargs)
            if len(gpu_uuids) != self.train_alloc.parallel.world_size:
                raise ValueError("GPU pool size must equal the v2 training world size")
            self.gpu_uuids = gpu_uuids
            self.frontend = frontend
            self.launch_id = uuid.uuid4().hex
            self._deferred_cleanup: Future | None = None
            self.deferred_processes: list[tuple[int, float]] = []

        def _finish_deferred_cleanup(self, _):
            try:
                # Cleanup removes scheduler records even if a process fails to exit.
                # Keep identities so a later stop still verifies every owned process.
                for role in self._service_roles:
                    for info in self.scheduler._workers.get(role, []):
                        if info.process is None:
                            continue
                        try:
                            parent = psutil.Process(info.process.pid)
                            self.deferred_processes.extend(
                                (p.pid, p.create_time())
                                for p in [parent, *parent.children(recursive=True)]
                            )
                        except psutil.NoSuchProcess:
                            pass
                self._cleanup_runtime_state()
            except BaseException as error:
                self._deferred_cleanup.set_exception(error)
            else:
                self._deferred_cleanup.set_result(None)

        def quiesce_initialization(self):
            """Stop resource creation before taking the final process snapshot."""
            self._shutdown_requested.set()
            future = self._init_future
            if future is not None:
                try:
                    future.result(timeout=self.config.setup_timeout)
                except FutureTimeoutError:
                    if not future.done():
                        if self._deferred_cleanup is None:
                            self._deferred_cleanup = Future()
                            future.add_done_callback(self._finish_deferred_cleanup)
                        raise RuntimeError(
                            "v2 initialization is still running; release cannot be verified. "
                            "Final cleanup will run when initialization settles."
                        ) from None
                except Exception:
                    # Initialization failures are already reported by initialize().
                    pass
                if self._deferred_cleanup is not None:
                    try:
                        self._deferred_cleanup.result(timeout=self.config.setup_timeout)
                    except FutureTimeoutError:
                        raise RuntimeError(
                            "v2 deferred cleanup is still running; release cannot be verified"
                        ) from None
                    except Exception as error:
                        raise RuntimeError(
                            "v2 deferred cleanup failed; release cannot be verified"
                        ) from error
                self._init_future = None

        def destroy(self):
            self.quiesce_initialization()
            super().destroy()

        async def _async_fork_on_guard(
            self,
            guard_addr,
            role,
            worker_index,
            raw_cmd,
            env=None,
            health_path="/health",
        ):
            if self.frontend is not None and role in {"router", "gateway"}:
                # Borrow stable services without adding them to this generation's
                # fork/role cleanup lists. Registration still uses a private key.
                return self.frontend.endpoints[role]
            child_env = dict(env or {})
            child_env["CUDA_VISIBLE_DEVICES"] = (
                self.gpu_uuids[worker_index] if role == "train-worker" else ""
            )
            child_env["AREAL_ELASTIC_LAUNCH_ID"] = self.launch_id
            return await super()._async_fork_on_guard(
                guard_addr, role, worker_index, raw_cmd, child_env, health_path
            )

        def rank_records(self, kind: str, step: int | None = None) -> list[dict]:
            root = Path(
                self.config.scheduling_spec[0].env_vars["AREAL_ELASTIC_DIAGNOSTICS_DIR"]
            )
            prefix = f"{self.launch_id}-{kind}"
            if step is not None:
                prefix += f"-{step:08d}"
            records = [
                json.loads((root / f"{prefix}-rank-{rank}.json").read_text())
                for rank in range(len(self.gpu_uuids))
            ]
            if any(record["rank"] != rank for rank, record in enumerate(records)):
                raise RuntimeError("v2 diagnostic records have inconsistent ranks")
            return records

        def service_endpoints(self) -> dict:
            return {
                "gateway": self._gateway_addr,
                "router": self._router_addr,
                "data_proxy": self._model_addr,
                "workers": list(self._worker_addrs),
            }

    # Retain the module-level class identity for repeat imports and serialization.
    LocalV2TrainController.__qualname__ = "LocalV2TrainController"
    globals()[name] = LocalV2TrainController
    return LocalV2TrainController
