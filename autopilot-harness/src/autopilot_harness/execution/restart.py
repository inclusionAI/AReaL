"""Restart a managed local training job from its acknowledged snapshot.

This layer executes restarts and previously validated configuration transitions.
It never makes new configuration decisions after pausing. One owner controls a run.
"""

import math
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from autopilot_harness.contracts.execution import ApplyPlan, ApplyResult
from autopilot_harness.monitor.outputs import HistoryStore

from .plan import check_apply_plan
from .process import LocalProcess
from .protocol import (
    BINDINGS,
    LAUNCH_ENV,
    check_message,
    check_snapshot,
    message,
    publish,
    read_json,
)


@dataclass(frozen=True)
class LaunchSpec:
    argv: tuple[str, ...]
    cwd: Path
    run_id: str
    runtime_revision: str
    configuration_id: str
    environment: dict[str, str] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if (
            isinstance(self.argv, str)
            or not self.argv
            or any(not isinstance(v, str) or not v or "\0" in v for v in self.argv)
        ):
            raise ValueError(
                "argv must be a nonempty sequence of arguments, not a shell command"
            )
        object.__setattr__(self, "argv", tuple(self.argv))
        object.__setattr__(self, "cwd", Path(self.cwd).resolve(strict=True))
        object.__setattr__(self, "environment", dict(self.environment))
        for key in ("run_id", "runtime_revision", "configuration_id"):
            if (
                not isinstance(getattr(self, key), str)
                or not getattr(self, key).strip()
            ):
                raise ValueError(f"LaunchSpec requires {key}")
        if LAUNCH_ENV in self.environment:
            raise ValueError("control launch environment is owned by the harness")


@dataclass(frozen=True)
class Timeouts:
    startup: float = 600
    snapshot: float = 1800
    exit: float = 120
    restore: float = 600
    poll: float = 0.05

    def __post_init__(self):
        for value in vars(self).values():
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError("timeouts must be positive finite seconds")


@dataclass
class RuntimeHandle:
    directory: Path
    launch: dict
    process: LocalProcess
    ready: dict | None = None


class RestartError(RuntimeError):
    def __init__(self, stage: str, directory: Path):
        super().__init__(
            f"runtime operation failed during {stage}; inspect {directory}"
        )
        self.stage = stage
        self.directory = directory


class RestartCoordinator:
    """Own an initial launch and subsequent cooperative restarts.

    start() -> restart() -> wait() runs one switch then lets training finish.
    restart() can be called again for a still-running resumed segment. close()
    explicitly terminates remaining owned groups; it never records apply success.
    On failure before replacement, context exit retains the source; inspect
    retained_source and explicitly close() only when cancellation is intended.
    """

    def __init__(
        self, spec: LaunchSpec, directory: Path, *, timeouts: Timeouts | None = None
    ):
        self.spec = spec
        self.directory = Path(directory).resolve()
        self.timeouts = timeouts or Timeouts()
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=False)
        (self.directory / "segments").mkdir()
        (self.directory / "snapshots").mkdir()
        (self.directory / "plans").mkdir()
        self.history = HistoryStore(self.directory / "events.jsonl")
        self.current: RuntimeHandle | None = None
        self.retained_source: RuntimeHandle | None = None
        self._owned = []
        self._lock = threading.Lock()
        self._failed = False
        self._closed = False
        self._configuration_id = spec.configuration_id
        self._configuration_file = None
        # Do not persist command arguments/env: they may carry credentials.
        publish(
            self.directory / "manifest.json",
            dict(
                schema_version=1,
                run_id=spec.run_id,
                runtime_revision=spec.runtime_revision,
                configuration_id=spec.configuration_id,
                backend="local_process_group",
                operation="managed_restart_and_apply",
            ),
        )

    def _event(self, kind, handle=None, **payload):
        event = dict(
            event_id=uuid4().hex,
            run_id=self.spec.run_id,
            segment_id=handle.launch["segment_id"] if handle else "harness",
            kind=kind,
            recorded_at=time.time(),
            payload=payload,
        )
        self.history.append(event)
        return event

    def _launch(self, restore=None, source_id=None, *, plan=None, target_file=None):
        segment_id = uuid4().hex
        directory = self.directory / "segments" / segment_id
        directory.mkdir(mode=0o700)
        launch = dict(
            schema_version=1,
            run_id=self.spec.run_id,
            segment_id=segment_id,
            nonce=uuid4().hex,
            runtime_revision=self.spec.runtime_revision,
            configuration_id=self._configuration_id
            if plan is None
            else plan.to_dict()["validation"]["candidate_id"],
            restore=restore,
            runtime_source_id=source_id,
        )
        launch["configuration_file"] = (
            self._configuration_file if plan is None else str(target_file)
        )
        launch["entry_configuration_id"] = self.spec.configuration_id
        if plan is not None:
            value = plan.to_dict()
            launch["plan_id"] = plan.plan_id
            launch["transition"] = dict(
                value["transition"],
                plan_id=plan.plan_id,
                source_configuration_id=value["validation"]["baseline_id"],
                target_configuration_id=value["validation"]["candidate_id"],
            )
        publish(directory / "launch.json", launch)
        environment = dict(os.environ, **self.spec.environment)
        environment[LAUNCH_ENV] = str(directory / "launch.json")
        process = LocalProcess(
            self.spec.argv, self.spec.cwd, environment, directory / "runtime.log"
        )
        handle = RuntimeHandle(directory, launch, process)
        self._owned.append(handle)
        self.current = handle
        self._event(
            "runtime.launched", handle, pid=process.pgid, restoring=restore is not None
        )
        return handle

    def _receipt(self, handle, kind):
        path = handle.directory / (kind + ".json")
        if not path.exists():
            return None
        return check_message(read_json(path), handle.launch, kind)

    def _until(self, handle, seconds, condition):
        deadline = time.monotonic() + seconds
        while True:
            failure = self._receipt(handle, "failed")
            if failure is not None:
                raise RuntimeError(f"runtime reported {failure.get('stage')}")
            result = condition()
            if result is not None:
                return result
            if handle.process.poll() is not None:
                raise RuntimeError("runtime exited before required receipt")
            if time.monotonic() >= deadline:
                raise TimeoutError("runtime acknowledgement timed out")
            time.sleep(self.timeouts.poll)

    def _wait_exit(self, handle):
        deadline = time.monotonic() + self.timeouts.exit
        while True:
            code = handle.process.poll()
            if code is not None and code != 0:
                raise RuntimeError("runtime exited unsuccessfully")
            if code == 0 and handle.process.group_gone():
                self._event("runtime.exited", handle, returncode=code, group_gone=True)
                return
            if time.monotonic() >= deadline:
                raise TimeoutError("runtime process group still exists")
            time.sleep(self.timeouts.poll)

    def _guard(self):
        if self._closed or self._failed:
            raise RuntimeError(
                "coordinator is closed or failed; inspect artifacts before recovery"
            )
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("another runtime operation is in progress")

    def _failure(self, stage, error):
        self._failed = True
        if stage in {"request_snapshot", "save_and_cleanup", "old_group_exit"}:
            self.retained_source = self.current
        self._event(
            "restart.failed", self.current, stage=stage, error_type=type(error).__name__
        )
        # Preserve an unsaved source. A failed replacement can be cleaned up:
        # its source snapshot and the old segment's exit evidence are durable.
        if stage in {"replacement_start", "restore"} and self.current is not None:
            try:
                self.current.process.terminate()
            except Exception as cleanup:
                self._event(
                    "runtime.cleanup_failed",
                    self.current,
                    error_type=type(cleanup).__name__,
                )
        return RestartError(stage, self.directory)

    def _check_completed(self, handle, completed, source_id):
        if (
            completed is None
            or completed.get("kind") != "completed"
            or completed.get("cleanup_completed") is not True
            or completed.get("status") != "completed"
            or completed.get("restored") != handle.launch["restore"]
            or completed.get("runtime_source_id") != source_id
        ):
            raise RuntimeError(
                "completion lacks matching restore and cleanup acknowledgement"
            )

    def start(self) -> RuntimeHandle:
        self._guard()
        try:
            if self.current is not None:
                raise RuntimeError("initial runtime already launched")
            handle = self._launch()
            ready = self._until(
                handle, self.timeouts.startup, lambda: self._receipt(handle, "ready")
            )
            if ready.get("status") != "initialized" or not ready.get(
                "runtime_source_id"
            ):
                raise RuntimeError("invalid initial readiness receipt")
            if handle.process.poll() is not None:
                raise RuntimeError("initial runtime already exited")
            handle.ready = ready
            self._event(
                "runtime.ready",
                handle,
                status="initialized",
                runtime_source_id=ready["runtime_source_id"],
            )
            return handle
        except Exception as error:
            raise self._failure("startup", error) from error
        finally:
            self._lock.release()

    def execution_context(self) -> dict:
        """Trusted live binding to pass to Validator, before requesting a stop."""
        old = self.current
        if old is None or old.ready is None or old.process.poll() is not None:
            raise RuntimeError("validation requires a live managed runtime")
        return dict(
            **{key: old.launch[key] for key in BINDINGS},
            runtime_source_id=old.ready["runtime_source_id"],
            apply_adapter=old.ready.get("apply_adapter"),
        )

    def apply(self, plan: ApplyPlan) -> ApplyResult:
        """Execute a passed plan; unchanged candidates do not pause training."""
        if not isinstance(plan, ApplyPlan):
            raise TypeError("apply requires an ApplyPlan")
        return self._restart(plan=plan)

    def restart(self) -> dict:
        return self._restart()

    def _restart(self, *, plan=None):
        self._guard()
        started = time.monotonic()
        target_file = None
        if plan is not None:
            # Everything here finishes before publishing the first stop request.
            # A rejected/stale plan leaves the running job and coordinator usable.
            try:
                value = check_apply_plan(plan, self.execution_context())
                validation = value["validation"]
                if validation["baseline_id"] == validation["candidate_id"]:
                    event = self._event(
                        "apply.unchanged", self.current, plan_id=plan.plan_id
                    )
                    result = ApplyResult(
                        "unchanged",
                        plan.plan_id,
                        validation["baseline_id"],
                        validation["candidate_id"],
                        self.current.launch["segment_id"],
                        self.current.launch["segment_id"],
                        event["event_id"],
                        event["recorded_at"],
                        time.monotonic() - started,
                    )
                    self._lock.release()
                    return result
                directory = (
                    self.directory / "plans" / plan.plan_id.removeprefix("sha256:")
                )
                directory.mkdir(mode=0o700)
                publish(directory / "plan.json", value)
                target_file = directory / "configuration.json"
                publish(target_file, value["candidate"])
                self._event(
                    "apply.validated",
                    self.current,
                    plan_id=plan.plan_id,
                    baseline_id=validation["baseline_id"],
                    candidate_id=validation["candidate_id"],
                    context_id=validation["context_id"],
                    execution_id=validation["execution_id"],
                )
            except Exception:
                self._lock.release()
                raise
        stage = "request_snapshot"
        try:
            old = self.current
            if old is None or old.ready is None or old.process.poll() is not None:
                raise RuntimeError("restart requires a live managed runtime")
            request_id = uuid4().hex
            request = message(
                old.launch,
                "stop_request",
                request_id=request_id,
                state_id=request_id,
                directory=str(self.directory / "snapshots" / request_id),
            )
            publish(old.directory / "stop_request.json", request)
            self._event("snapshot.requested", old, request_id=request_id)
            stage = "save_and_cleanup"
            stopped = self._until(
                old, self.timeouts.snapshot, lambda: self._receipt(old, "stopped")
            )
            if (
                stopped.get("request_id") != request_id
                or stopped.get("cleanup_completed") is not True
            ):
                raise RuntimeError("stop receipt lacks request/cleanup confirmation")
            source = check_snapshot(stopped, old.launch, request)
            self._event("snapshot.saved", old, request_id=request_id, **source)
            stage = "old_group_exit"
            self._wait_exit(old)
            stage = "replacement_start"
            new = self._launch(
                source,
                old.ready["runtime_source_id"],
                plan=plan,
                target_file=target_file,
            )
            stage = "restore"
            receipt = self._until(
                new,
                self.timeouts.restore,
                lambda: self._receipt(new, "ready") or self._receipt(new, "completed"),
            )
            if (
                receipt.get("restored") != source
                or receipt.get("runtime_source_id") != old.ready["runtime_source_id"]
                or receipt.get("status") not in {"resumed", "completed"}
                or (
                    plan is not None
                    and receipt.get("apply_adapter")
                    != value["transition"]["adapter_id"]
                )
            ):
                raise RuntimeError(
                    "restore acknowledgement does not match source/runtime"
                )
            if receipt["status"] == "completed":
                self._check_completed(new, receipt, old.ready["runtime_source_id"])
                self._wait_exit(new)
            elif new.process.poll() is not None:
                completed = self._receipt(new, "completed")
                self._check_completed(new, completed, old.ready["runtime_source_id"])
                self._wait_exit(new)
                receipt = completed
            new.ready = receipt
            if plan is not None:
                self._configuration_id = new.launch["configuration_id"]
                self._configuration_file = str(target_file)
                event = self._event(
                    "apply.confirmed",
                    new,
                    plan_id=plan.plan_id,
                    source_segment=old.launch["segment_id"],
                    snapshot=source["snapshot"],
                    baseline_id=validation["baseline_id"],
                    configuration_id=self._configuration_id,
                    status=receipt["status"],
                    next_step=source["context"]["step_info"]["global_step"] + 1,
                    elapsed_seconds=time.monotonic() - started,
                    exposed_delay_seconds=None,
                )
                result = ApplyResult(
                    "applied",
                    plan.plan_id,
                    validation["baseline_id"],
                    self._configuration_id,
                    old.launch["segment_id"],
                    new.launch["segment_id"],
                    event["event_id"],
                    event["recorded_at"],
                    time.monotonic() - started,
                    source["context"]["step_info"]["global_step"] + 1,
                    source["snapshot"]["path"],
                )
                publish(new.directory / "apply_result.json", result.to_dict())
                return result
            event = self._event(
                "restart.confirmed",
                new,
                source_segment=old.launch["segment_id"],
                snapshot=source["snapshot"],
                status=receipt["status"],
                next_step=source["context"]["step_info"]["global_step"] + 1,
            )
            result = dict(
                operation="same_configuration_restart",
                status=receipt["status"],
                source_segment=old.launch["segment_id"],
                segment_id=new.launch["segment_id"],
                snapshot=source["snapshot"],
                context=source["context"],
                event_id=event["event_id"],
            )
            publish(new.directory / "restart_result.json", result)
            return result
        except Exception as error:
            if plan is not None:
                self._event(
                    "apply.failed",
                    self.current,
                    plan_id=plan.plan_id,
                    stage=stage,
                    baseline_id=validation["baseline_id"],
                    candidate_id=validation["candidate_id"],
                    error_type=type(error).__name__,
                    elapsed_seconds=time.monotonic() - started,
                    exposed_delay_seconds=None,
                )
            raise self._failure(stage, error) from error
        finally:
            self._lock.release()

    def wait(self, *, timeout: float = 86400) -> dict:
        """Wait for normal completion and group exit of the current runtime."""
        if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("completion timeout must be positive and finite")
        self._guard()
        try:
            if self.current is None:
                raise RuntimeError("no runtime launched")
            completed = self._until(
                self.current, timeout, lambda: self._receipt(self.current, "completed")
            )
            self._check_completed(
                self.current, completed, self.current.ready["runtime_source_id"]
            )
            self._wait_exit(self.current)
            self._event("runtime.completed", self.current)
            return completed
        except Exception as error:
            raise self._failure("completion", error) from error
        finally:
            self._lock.release()

    def close(self):
        if self._closed:
            return
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("cannot close during another runtime operation")
        errors = []
        try:
            for handle in reversed(self._owned):
                try:
                    handle.process.terminate()
                except Exception as error:
                    errors.append(error)
            self._closed = not errors
            self._failed = self._failed or bool(errors)
        finally:
            self._lock.release()
        if errors:
            raise RuntimeError("one or more owned process groups remain") from errors[0]

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        if self.retained_source is not None and not self._closed:
            # A snapshot timeout is not permission to kill unsaved training.
            # Explicit close() remains available to cancel the owned source.
            self._event(
                "runtime.retained",
                self.retained_source,
                pid=self.retained_source.process.pgid,
                reason="restart_failed_before_replacement",
            )
            return
        try:
            self.close()
        except Exception:
            if value is None:
                raise
            self._event("runtime.cleanup_failed", self.current)
