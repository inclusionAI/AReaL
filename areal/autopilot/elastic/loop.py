# SPDX-License-Identifier: Apache-2.0

"""Execute resource decisions while preserving training/publication consistency."""

import hashlib
import math
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path

from areal.autopilot.contracts import (
    Decision,
    ElasticObservation,
    HistoryWindow,
    Measurement,
    ResourceDecision,
    SchedulingPolicy,
)
from areal.autopilot.elastic.policy.basic.joint import JointConfig
from areal.autopilot.elastic.policy.basic.online import JointResourceDecision
from areal.autopilot.elastic.policy.basic.policy import (
    IdleTimeoutPolicy,
)
from areal.autopilot.elastic.runtime.config import ConfigApplication, ConfigRequest
from areal.autopilot.elastic.runtime.history import RuntimeHistory
from areal.autopilot.elastic.runtime.policy import (
    PolicyApplication,
    PolicyBoundaryBusy,
    PolicyRequest,
)
from areal.autopilot.elastic.runtime.state.checkpoint import atomic_json
from areal.autopilot.elastic.runtime.training.trainer import QueuedTrainer


class ElasticLoop:
    """Single lifecycle thread and one gated producer; strict current-policy data.

    Publication failures retry the same committed artifact. An uncertain optimizer
    or checkpoint outcome is fatal until stop/prepare recovers the durable cursor.
    External producers bypassing episode() are unsupported.
    """

    def __init__(
        self,
        trainer: QueuedTrainer,
        publisher,
        *,
        policy: SchedulingPolicy | None = None,
        clock: Callable[[], float] = time.monotonic,
        journal: Path | None = None,
        config_application: ConfigApplication | None = None,
        policy_bundle_id: str = "constructor-policy",
        policy_application: PolicyApplication | None = None,
        runtime_history: RuntimeHistory | None = None,
    ):
        self._policy = IdleTimeoutPolicy() if policy is None else policy
        if not callable(getattr(self._policy, "decide", None)):
            raise TypeError("Scheduling policy must implement decide()")
        self.trainer, self.publisher = trainer, publisher
        self.clock, self.journal = clock, journal
        self.config_application = config_application
        if not isinstance(policy_bundle_id, str) or not policy_bundle_id:
            raise ValueError("Scheduling policy provenance is required")
        self.policy_bundle_id = policy_bundle_id
        self.policy_application = policy_application
        self.policy_generation = 0
        self.history = getattr(self._policy, "history", HistoryWindow())
        self._lifecycle_lock = threading.RLock()
        self._policy_failed = False
        self.runtime_history = runtime_history
        if callable(getattr(self._policy, "decide_joint", None)) and (
            config_application is None
            or runtime_history is None
            or policy_application is not None
            and policy_application.execution_context.get("scope")
            != "serial-joint-timing-v1"
        ):
            raise ValueError(
                "Online joint planning requires config application and durable history; "
                "fixed-Rollout revision bundles have a different execution scope"
            )
        self._episode_sample = None
        self._update_observation = None
        self._update_measurements = []
        self._planning_history = None
        self._planning_joint_window = None
        self.last_config_receipt = None
        self._batch_ready = self._gate_available = False
        self.state = "NEW"
        self.pending_checkpoint = None
        self.published_version = None
        self.events = []
        self.last_activity = clock()
        self._episode_started = None
        self._last_workload = None
        self._last_decision_record = None
        self._refresh_observation()

    def _refresh_observation(self) -> None:
        now = self.clock()
        max_train_gpus = self.trainer.max_train_gpus
        if self.config_application is not None:
            max_train_gpus = getattr(
                self.config_application.executor, "max_train_gpus", max_train_gpus
            )
        self._observation = ElasticObservation(
            state=self.state,
            model_version=self.trainer.version,
            serving_version=self.published_version,
            training_active=self.trainer.is_active,
            batch_ready=self._batch_ready,
            idle_duration=max(0, now - self.last_activity),
            gate_available=self._gate_available,
            pending_publication=self.pending_checkpoint is not None,
            committed=self.trainer.is_committed,
            observed_at=now,
            training_config=self.trainer.training_config,
            next_batch=self.trainer.next_workload() if self._batch_ready else None,
            committed_samples=self.trainer.committed_samples,
            committed_tokens=self.trainer.committed_tokens,
            max_train_gpus=max_train_gpus,
            has_checkpoint=self.trainer.latest_checkpoint is not None,
            episode_active=self._episode_started is not None,
            episode_elapsed=0
            if self._episode_started is None
            else max(0, now - self._episode_started),
        )

    def observe(self) -> ElasticObservation:
        """Return the last immutable lifecycle sample; never inspect live workers."""
        return self._observation

    def _decide(self) -> Decision | ResourceDecision:
        self._refresh_observation()
        self._planning_history = asdict(self.history.snapshot())
        self._planning_joint_window = None
        joint = getattr(self._policy, "decide_joint", None)
        if callable(joint) and (
            self._batch_ready
            and self._gate_available
            and self.trainer.is_committed
            and self.pending_checkpoint is None
        ):
            executor = self.config_application.executor
            decision = joint(
                self.observe(),
                executor.observe().config,
                len(self.trainer.group.pool_devices),
            )
            if self._policy.last_window is not None:
                self._planning_joint_window = asdict(self._policy.last_window)
        else:
            decision = self._policy.decide(self.observe())
        if isinstance(decision, ResourceDecision):
            return decision
        if decision not in {"train", "keep", "release"}:
            raise ValueError(f"Invalid resource decision: {decision!r}")
        return decision

    def _record(self, event: str, **extra):
        self._refresh_observation()
        self.events.append(
            {
                "event": event,
                "time": self.clock(),
                "step": self.trainer.version,
                "policy_bundle_id": self.policy_bundle_id,
                "policy_generation": self.policy_generation,
                **extra,
            }
        )
        if self.journal is not None:
            atomic_json(self.journal, self.events)

    def start(self) -> None:
        with self.publisher.lock:
            if self.state != "NEW":
                raise RuntimeError("Elastic loop has already started")
            try:
                self.trainer.prepare()
                if self.runtime_history is not None:
                    execution = self.runtime_history.execution
                    if (
                        execution["queue_id"] != self.trainer.queue.queue_id
                        or execution["checkpoint_root"]
                        != str(self.trainer.group.store.root.resolve())
                        or execution["compatibility"]
                        != self.trainer.group.compatibility
                    ):
                        raise ValueError(
                            "Runtime history does not match this execution"
                        )
                    self.runtime_history.validate_version(self.trainer.version)
                if self.policy_application is not None:
                    application = self.policy_application
                    execution = application.execution_context
                    if (
                        application.initial_bundle_id != self.policy_bundle_id
                        or application.compatibility != self.trainer.group.compatibility
                        or execution["queue_id"] != self.trainer.queue.queue_id
                        or execution["checkpoint_root"]
                        != str(self.trainer.group.store.root.resolve())
                    ):
                        raise ValueError(
                            "Policy application does not match this execution"
                        )
                    if execution.get("scope") == "serial-joint-timing-v1" and (
                        self.config_application is None
                        or execution["joint_executor"]
                        != self.config_application.executor.context_id
                        or execution["max_total_gpus"]
                        != len(self.trainer.group.pool_devices)
                    ):
                        raise ValueError(
                            "Joint policy application has a different executor/pool"
                        )
                    prepared = application.recover()
                    self.history = application.history(self.trainer.version)
                    if prepared is not None:
                        self._policy = prepared.policy
                    if hasattr(self._policy, "history"):
                        self._policy.history = self.history
                    state = application.state()
                    self.policy_bundle_id = state["bundle_id"]
                    self.policy_generation = state["generation"]
                self._hydrate_policy(
                    self._policy,
                    self.history if self.policy_application is not None else None,
                )
                if self.config_application is not None:
                    validate = getattr(self._policy, "validate_executor", None)
                    if callable(validate):
                        validate(self.config_application.executor)
                    self._config_hook("prepare")
                    self.config_application.recover()
                self.state = "ACTIVE" if self.trainer.is_active else "SUSPENDED"
                self._batch_ready = self.trainer.has_pending_batch()
                self._gate_available = True
                if self.trainer.latest_checkpoint is None:
                    self.publisher.initialize()
                    self.published_version = 0
                else:
                    self.pending_checkpoint = self.trainer.latest_checkpoint
                    self._publish_pending()
                if self.pending_checkpoint is None:
                    self._config_hook("resume")
                self._record("PREPARED")
            except BaseException:
                self.publisher.ready = False
                self.state = "FAILED"
                self._refresh_observation()
                raise

    def _hydrate_policy(self, policy, history=None):
        restore = getattr(policy, "restore_runtime", None)
        if callable(restore):
            if self.runtime_history is None:
                raise ValueError("Joint policy requires durable runtime history")
            state = None
            if policy is not self._policy and callable(
                getattr(self._policy, "runtime_state", None)
            ):
                state = self._policy.runtime_state()
            elif self.policy_application is not None:
                state = self.policy_application.joint_state(self.trainer.version)
            snapshot = self.runtime_history.snapshot(self.trainer.version)
            if state is None:
                restore(snapshot, history=history)
            else:
                if state["through_version"] > self.trainer.version:
                    raise ValueError("Joint runtime state is newer than weights")
                policy.restore_joint_state(state, history=history)
                for row in snapshot["updates"]:
                    policy.record_committed(row)
                for row in snapshot["boundaries"]:
                    policy.record_boundary(row)
            if policy is self._policy:
                self.history = policy.history
        elif history is not None:
            policy.history = history

    def prepare_policy(self, bundle: Path):
        """Load trusted factories on CPU before taking the execution boundary."""
        if self.policy_application is None:
            raise RuntimeError("No policy application store is installed")
        prepared = self.policy_application.prepare(bundle)
        snapshot = self.history.snapshot()
        shadow = HistoryWindow(
            max(32, len(snapshot.workloads), (len(snapshot.measurements) + 15) // 16)
        )
        for row in snapshot.workloads:
            shadow.add_workload(row)
        for row in snapshot.measurements:
            shadow.add_measurement(row)
        self._hydrate_policy(prepared.policy, shadow)
        validate = getattr(prepared.policy, "validate_executor", None)
        if callable(validate):
            if self.config_application is None:
                raise ValueError("Joint policy needs a configuration executor")
            validate(self.config_application.executor)
        decision = prepared.policy.decide(self.observe())
        if not isinstance(decision, ResourceDecision) and decision not in {
            "keep",
            "train",
            "release",
        }:
            raise ValueError("Prepared policy returned an invalid decision")
        return replace(prepared, probed=True)

    def _policy_boundary(self):
        if (
            self.state in {"NEW", "CLOSED"}
            or self._episode_started is not None
            or self.pending_checkpoint is not None
            or not self.trainer.is_committed
            or self.published_version != self.trainer.version
            or self.config_application is not None
            and self.config_application.state()["pending"] is not None
        ):
            raise PolicyBoundaryBusy(
                "Policy application needs a committed planning boundary"
            )

    def apply_policy(self, request: PolicyRequest, prepared) -> dict:
        """Commit one accepted bundle without restarting workers or changing weights."""
        if self.policy_application is None:
            raise RuntimeError("No policy application store is installed")
        with self._lifecycle_lock:
            # Do not wait through a long episode while preventing idle GPU release.
            if not self.publisher.lock.acquire(blocking=False):
                raise PolicyBoundaryBusy(
                    "Policy application must wait for the episode boundary"
                )
            try:
                self._policy_boundary()
                if self.state == "FAILED" or not self.publisher.ready:
                    raise RuntimeError("Policy application needs ready serving weights")
                application = self.policy_application
                receipt = application.preflight(request, self.trainer.version)
                if receipt is not None:
                    return receipt
                if (
                    prepared.context_id != application.context
                    or prepared.bundle_id != request.bundle_id
                    or not prepared.probed
                ):
                    raise ValueError("Prepared policy does not match the request")
                candidate = prepared.policy
                self._hydrate_policy(candidate, self.history)
                self.publisher.ready = False
                try:
                    receipt = application.commit(
                        request,
                        self.trainer.version,
                        prepared,
                        self.history,
                        joint_state=self._joint_state(candidate),
                    )
                    candidate.history = self.history
                    candidate.last_search = None
                    self._policy = candidate
                    self.policy_bundle_id = request.bundle_id
                    self.policy_generation = receipt["generation"]
                    self._last_decision_record = None
                    self._record("POLICY_APPLIED", receipt=receipt)
                    self.publisher.ready = True
                    return receipt
                except BaseException:
                    # A failed fsync can occur after the active pointer was replaced.
                    self._policy_failed = True
                    self.publisher.ready = False
                    self.state = "FAILED"
                    self._refresh_observation()
                    raise
            finally:
                self.publisher.lock.release()

    def recover_policy(self):
        """Follow the durable policy pointer; never replay a train/config operation."""
        if self.policy_application is None or not self._policy_failed:
            raise RuntimeError("No policy activation failure is pending")
        with self._lifecycle_lock:
            if not self.publisher.lock.acquire(blocking=False):
                raise RuntimeError("Policy recovery must wait for the episode boundary")
            try:
                self._policy_boundary()
                self.publisher.ready = False
                prepared = self.policy_application.recover()
                if prepared is not None:
                    self._hydrate_policy(prepared.policy, self.history)
                    self._policy = prepared.policy
                state = self.policy_application.state()
                self.policy_bundle_id, self.policy_generation = (
                    state["bundle_id"],
                    state["generation"],
                )
                self._last_decision_record = None
                self.state = "ACTIVE" if self.trainer.is_active else "SUSPENDED"
                self._record("POLICY_RECOVERED")
                self._policy_failed = False
                self.publisher.ready = True
            except BaseException:
                self._policy_failed = True
                self.publisher.ready = False
                self.state = "FAILED"
                self._refresh_observation()
                raise
            finally:
                self.publisher.lock.release()

    @staticmethod
    def _joint_state(policy):
        snapshot = getattr(policy, "runtime_state", None)
        return snapshot() if callable(snapshot) else None

    def _save_history(self):
        if self.policy_application is not None:
            self.policy_application.save_history(
                self.history,
                self.trainer.version,
                joint_state=self._joint_state(self._policy),
            )

    @contextmanager
    def _execution_gate(self):
        with self._lifecycle_lock:
            if not self.publisher.lock.acquire(blocking=False):
                raise RuntimeError(
                    "Configuration application must wait for the episode boundary"
                )
            try:
                yield
            finally:
                self.publisher.lock.release()

    def apply_config(self, request: ConfigRequest) -> dict:
        """Explicit config application under the same gate as episodes and updates."""
        if self.config_application is None:
            raise RuntimeError("No configuration application executor is installed")
        with self._execution_gate():
            self._validate_policy_config(request)
            if (
                self.state in {"NEW", "FAILED", "CLOSED"}
                or self._episode_started is not None
                or self.pending_checkpoint is not None
                or not self.trainer.is_committed
                or not self.publisher.ready
                or self.published_version != self.trainer.version
            ):
                raise RuntimeError(
                    "Configuration application needs a committed boundary"
                )
            # Unsupported targets leave admission and workers unchanged.
            try:
                self.config_application.preflight(request)
            except (ValueError, TypeError):
                raise
            except Exception:
                self.publisher.ready = False
                self.state = "FAILED"
                self._refresh_observation()
                raise
            self.publisher.ready = False
            try:
                receipt = self.config_application.apply(request)
                self._config_hook("resume")
                self.last_config_receipt = receipt
                self.state = "ACTIVE" if self.trainer.is_active else "SUSPENDED"
                self._record("CONFIG_APPLIED", receipt=receipt)
                self.publisher.ready = True
                return receipt
            except BaseException:
                self.publisher.ready = False
                self.state = "FAILED"
                self._refresh_observation()
                raise

    def recover_config(self) -> dict | None:
        """Resolve a config failure without replaying a training update."""
        if self.config_application is None:
            raise RuntimeError("No configuration application executor is installed")
        with self._execution_gate():
            if self._policy_failed:
                raise RuntimeError("Recover the pending policy activation first")
            if (
                self.state in {"NEW", "CLOSED"}
                or self._episode_started is not None
                or self.pending_checkpoint is not None
            ):
                raise RuntimeError("Cannot recover configuration at this boundary")
            self.publisher.ready = False
            try:
                receipt = self.config_application.recover()
                if self.published_version != self.trainer.version:
                    raise RuntimeError(
                        "Recovered configuration has unpublished weights"
                    )
                self._config_hook("resume")
                self.last_config_receipt = receipt
                self.state = "ACTIVE" if self.trainer.is_active else "SUSPENDED"
                self._record("CONFIG_RECOVERED", receipt=receipt)
                self.publisher.ready = True
                return receipt
            except BaseException:
                self.publisher.ready = False
                self.state = "FAILED"
                self._refresh_observation()
                raise

    def _config_hook(self, name):
        if self.config_application is not None:
            callback = getattr(self.config_application.executor, name, None)
            if callback is not None:
                callback()

    def _config_request(self, target):
        state = self.config_application.state()
        if not isinstance(target, JointConfig):
            target = JointConfig(
                target, self.config_application.executor.observe().config.rollout
            )
        identity = hashlib.sha256(repr(target).encode()).hexdigest()[:12]
        request = ConfigRequest(
            f"decision-{self.trainer.version}-{state['generation']}-{identity}",
            target,
            state["generation"],
            self.trainer.version,
            self.policy_bundle_id,
        )
        self._validate_policy_config(request)
        return request

    def _validate_policy_config(self, request):
        if self.policy_application is None:
            return
        if request.policy_bundle_id != self.policy_bundle_id:
            raise ValueError("Config request belongs to another scheduling policy")
        execution = self.policy_application.execution_context
        if execution.get("scope") == "serial-joint-timing-v1":
            if request.target not in self._policy.optimizer.candidates:
                raise ValueError("Configuration is outside the joint policy allowlist")
            return
        fixed = execution.get("rollout")
        if (
            fixed is not None
            and asdict(request.target.rollout) != fixed
            or request.target.rollout.world_size
            != self.policy_application.execution_context["inference_gpus"]
        ):
            raise ValueError("Configuration changes the policy's fixed Rollout context")

    def _configure(self, target, request=None):
        if self.config_application is None:
            return self.trainer.configure(target)
        request = request or self._config_request(target)
        replayed = request.request_id in self.config_application.state()["receipts"]
        receipt = self.config_application.apply(request)
        self.last_config_receipt = receipt
        self._record("CONFIG_APPLIED", receipt=receipt)
        self.trainer.last_reconfiguration_error = (
            receipt["error"] if receipt["status"] == "rolled_back" else None
        )
        if receipt["request"]["target"]["actor"] != asdict(target):
            raise RuntimeError(
                "Configuration receipt differs from the lifecycle decision"
            )
        return (
            ()
            if replayed
            else getattr(self.config_application.executor, "measurements", ())
        )

    @contextmanager
    def episode(self):
        """Keep enqueue inside this gate; one outstanding episode per version."""
        with self.publisher.episode() as version:
            if (
                self.state in {"NEW", "FAILED", "CLOSED"}
                or self.pending_checkpoint is not None
                or self.trainer.has_pending_batch()
                or version != self.trainer.version
                or version != self.published_version
            ):
                raise RuntimeError(
                    "Previous episode is still awaiting consumption/publication"
                )
            episode_config = (
                asdict(self.config_application.executor.observe().config)
                if callable(getattr(self._policy, "decide_joint", None))
                else None
            )
            self._episode_started = self.clock()
            try:
                yield version
                workload = self.trainer.next_workload()
                if workload is not None:
                    measured = replace(
                        workload,
                        rollout_seconds=max(0, self.clock() - self._episode_started),
                    )
                    if hasattr(self._policy, "record_workload"):
                        self._policy.record_workload(measured)
                    if getattr(self._policy, "history", None) is not self.history:
                        self.history.add_workload(measured)
                    if callable(getattr(self._policy, "decide_joint", None)):
                        self._episode_sample = {
                            "joint_config": episode_config,
                            "rollout": episode_config["rollout"],
                            "workload": asdict(measured),
                            "seconds": measured.rollout_seconds,
                        }
            finally:
                self._episode_started = None

    def _measure(self, phase, started, workload=None):
        self._record_phase(
            phase,
            self.trainer.training_config,
            max(0, self.clock() - started),
            workload,
        )

    def _record_phase(self, phase, config, seconds, workload=None):
        boundary = getattr(self._policy, "record_boundary", None)
        if (
            phase == "release"
            and self._update_observation is None
            and self.runtime_history is not None
            and callable(boundary)
        ):
            boundary(
                self.runtime_history.append_boundary(
                    {
                        "weights_version": self.trainer.version,
                        "phase": phase,
                        "config": asdict(config),
                        "joint_config": asdict(
                            self.config_application.executor.observe().config
                        ),
                        "seconds": seconds,
                        "sequence_tokens": 0
                        if workload is None
                        else workload.sequence_tokens,
                    }
                )
            )
        if self._update_observation is not None and self.runtime_history is not None:
            self._update_measurements.append(
                {
                    "phase": phase,
                    "config": None if config is None else asdict(config),
                    "seconds": seconds,
                    "workload": None if workload is None else asdict(workload),
                }
            )
        if hasattr(self._policy, "record_phase"):
            self._policy.record_phase(phase, config, seconds, workload)
        if (
            config is not None
            and getattr(self._policy, "history", None) is not self.history
        ):
            self.history.add_measurement(
                Measurement(
                    phase,
                    config.world_size,
                    seconds,
                    0 if workload is None else workload.sequence_tokens,
                    config.profile_id,
                )
            )

    def _publish_pending(self) -> bool:
        started = self.clock()
        self.state = "PUBLISHING"
        self._refresh_observation()
        try:
            receipt = self.publisher.publish(self.pending_checkpoint)
            if not self.publisher.ready or receipt["version"] != self.trainer.version:
                raise RuntimeError(
                    "Publication has not completed at the committed step"
                )
        except Exception as error:
            self.publisher.ready = False
            self._record("PUBLISH_RETRY", error=f"{type(error).__name__}: {error}")
            return False
        self._measure("publish", started, self._last_workload)
        self.pending_checkpoint = None
        self.published_version = receipt["version"]
        self.last_activity = self.clock()
        self.state = "ACTIVE" if self.trainer.is_active else "SUSPENDED"
        self._record("PUBLISHED", seconds=self.clock() - started, receipt=receipt)
        if self.runtime_history is not None and self._update_observation is not None:
            row = {
                "weights_version": self.trainer.version,
                "checkpoint_id": self.trainer.latest_checkpoint.name,
                "policy_bundle_id": self.policy_bundle_id,
                "policy_generation": self.policy_generation,
                "observation": self._update_observation,
                "history": self._planning_history,
                "measurements": self._update_measurements,
            }
            if self.config_application is not None:
                row["joint_config"] = asdict(
                    self.config_application.executor.observe().config
                )
                receipt = self.last_config_receipt
                if (
                    receipt is not None
                    and receipt["request"]["expected_weights_version"]
                    == self._update_observation["model_version"]
                ):
                    row["config_receipt"] = receipt
            if self._planning_joint_window is not None:
                row["joint_window"] = self._planning_joint_window
            if self._episode_sample is not None:
                row["episode"] = self._episode_sample
            self.runtime_history.append(row)
            record = getattr(self._policy, "record_committed", None)
            if callable(record):
                record(row)
        self._save_history()
        self._episode_sample = None
        self._update_observation, self._update_measurements = None, []
        return True

    def tick(self) -> str:
        with self._lifecycle_lock:
            return self._tick()

    def _tick(self) -> str:
        if self.state in {"NEW", "FAILED", "CLOSED"}:
            raise RuntimeError("Elastic loop is not running")
        # An inference episode may hold the gate for minutes. Clean GPU release
        # changes neither serving weights nor cursor, so it need not wait for it.
        acquired = self.publisher.lock.acquire(blocking=False)
        self._gate_available = acquired
        try:
            try:
                if self.pending_checkpoint is not None:
                    if acquired:
                        self._publish_pending()
                    return self.state
                self._batch_ready = self.trainer.has_pending_batch()
                decision = self._decide()
                resource = decision if isinstance(decision, ResourceDecision) else None
                if resource is not None:
                    search = getattr(self._policy, "last_search", None)
                    joint = isinstance(resource, JointResourceDecision)
                    decision_key = (resource, self.trainer.version)
                    if decision_key != self._last_decision_record:
                        self._record(
                            "DECIDED",
                            action=resource.action,
                            config=None
                            if resource.config is None
                            else resource.config.world_size,
                            profile_id=None
                            if resource.config is None
                            else resource.config.profile_id,
                            train=resource.train,
                            reason=resource.reason,
                            predicted_gpu_seconds=None
                            if search is None or search.prediction is None
                            else search.prediction.total_gpu_seconds,
                            predicted_training_gpu_seconds=None
                            if search is None or search.prediction is None
                            else getattr(
                                search.prediction,
                                "training_gpu_seconds",
                                getattr(search.prediction, "actor_gpu_seconds", None),
                            ),
                            predicted_seconds=None
                            if search is None or search.prediction is None
                            else search.prediction.seconds,
                            evaluated=0
                            if search is None
                            else getattr(
                                search,
                                "evaluated",
                                len(getattr(search, "candidates", ())),
                            ),
                            joint_config=asdict(resource.target) if joint else None,
                            history_through_version=getattr(
                                self._policy, "through_version", None
                            ),
                            committed_samples=self.trainer.committed_samples,
                            effective_tokens=self.trainer.committed_tokens,
                        )
                        self._last_decision_record = decision_key
                    invalid = (
                        (resource.action == "launch" and self.trainer.is_active)
                        or (
                            resource.action == "reconfigure"
                            and not self.trainer.is_active
                        )
                        or (
                            resource.action == "keep"
                            and resource.config is not None
                            and (
                                resource.target
                                != self.config_application.executor.observe().config
                                if joint
                                else resource.config != self.trainer.training_config
                            )
                        )
                    )
                    if invalid:
                        self._record("DECISION_REJECTED", decision=resource.action)
                        return self.state
                    if resource.action == "release":
                        self._release()
                        return self.state
                    if not resource.train:
                        return self.state
                    if resource.config is not None:
                        try:
                            validator = getattr(self.trainer, "validate_config", None)
                            if validator is not None:
                                validator(resource.config)
                            elif resource.config.profile_id:
                                raise ValueError(
                                    "Training executor does not validate named actor profiles"
                                )
                        except (ValueError, TypeError, RuntimeError) as error:
                            self._record(
                                "DECISION_REJECTED",
                                decision=resource.action,
                                reason=str(error),
                                profile_id=resource.config.profile_id,
                            )
                            return self.state
                    decision = "train"
                if decision == "keep":
                    return self.state
                if decision == "release":
                    self._release()
                    return self.state
                if (
                    not acquired
                    or not self._batch_ready
                    or not self.trainer.is_committed
                ):
                    self._record("DECISION_REJECTED", decision=decision)
                    return self.state
                batch_id = self.trainer.next_batch_id()
                if batch_id is None:
                    raise RuntimeError("Selected queue batch disappeared")
                request = None
                joint_target = (
                    resource.target
                    if isinstance(resource, JointResourceDecision)
                    else None
                )
                config_changed = (
                    joint_target != self.config_application.executor.observe().config
                    if joint_target is not None
                    else resource is not None
                    and resource.config is not None
                    and resource.config != self.trainer.training_config
                )
                if self.config_application is not None and config_changed:
                    request = self._config_request(joint_target or resource.config)
                    try:
                        self.config_application.preflight(request)
                    except (ValueError, TypeError) as error:
                        self._record(
                            "DECISION_REJECTED",
                            decision=resource.action,
                            reason=str(error),
                        )
                        return self.state
                # Serialize selection, update and publication with whole episodes.
                self.publisher.ready = False
                self._update_observation = asdict(self.observe())
                self._update_measurements = []
                self._last_workload = self.trainer.next_workload()
                if config_changed:
                    measurements = self._configure(resource.config, request)
                    if self.trainer.last_reconfiguration_error is not None:
                        self._record(
                            "RECONFIGURE_ROLLED_BACK",
                            error=self.trainer.last_reconfiguration_error,
                            actual_config=self.trainer.training_config.world_size,
                        )
                    for phase, config, seconds in measurements:
                        self._record_phase(phase, config, seconds, self._last_workload)
                if not self.trainer.is_active:
                    self.state = "STARTING"
                    self._record("WAKING", batch_id=batch_id)
                    started = self.clock()
                    self.trainer.activate()
                    self._measure(
                        "restore"
                        if self.trainer.latest_checkpoint is not None
                        else "start",
                        started,
                        self._last_workload,
                    )
                    self._record("READY", seconds=self.clock() - started)
                self.state = "ACTIVE"
                self._refresh_observation()
                started = self.clock()
                result = self.trainer.train_next()
                if result is None:
                    raise RuntimeError("Selected queue batch disappeared")
                self._measure("train", started, self._last_workload)
                self._record("UPDATED", seconds=self.clock() - started, result=result)
                started = self.clock()
                self.pending_checkpoint = self.trainer.checkpoint()
                self._measure("checkpoint", started, self._last_workload)
                self._batch_ready = self.trainer.has_pending_batch()
                self._publish_pending()
                return self.state
            except BaseException:
                self.publisher.ready = False
                self.state = "FAILED"
                raise
        finally:
            try:
                self._refresh_observation()
            finally:
                if acquired:
                    self.publisher.lock.release()

    def _release(self) -> None:
        if (
            self.state != "ACTIVE"
            or self.pending_checkpoint is not None
            or not self.trainer.is_committed
            or self.trainer.has_pending_batch()
        ):
            self._record("DECISION_REJECTED", decision="release")
            return
        self.state = "STOPPING"
        started = self.clock()
        self._record("RELEASING")
        self.trainer.suspend()
        self._measure("release", started, self._last_workload)
        self.state = "SUSPENDED"
        self._record("RELEASED", seconds=self.clock() - started)
        self._save_history()

    def run(self, stop: threading.Event, *, poll_seconds: float = 1) -> None:
        if not math.isfinite(poll_seconds) or poll_seconds <= 0:
            raise ValueError("Poll interval must be finite and positive")
        while not stop.is_set():
            self.tick()
            stop.wait(poll_seconds)

    def close(self) -> None:
        """Call after run() exits; cancellation alone cannot stop a lifecycle thread."""
        with self.publisher.lock:
            self.publisher.ready = False
            self._config_hook("close")
            self.trainer.stop()
            self.state = "CLOSED"
            self._refresh_observation()
