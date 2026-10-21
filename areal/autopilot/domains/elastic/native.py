# SPDX-License-Identifier: Apache-2.0

"""Policy observation, durable history and accepted activation for native PPO.

This adapter never creates workers or executes PPO. Native async measurements are
conditional service times; they do not establish makespan or GPU savings.
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Any

from areal.autopilot.domains.elastic.contracts import workflow_from_profile
from areal.autopilot.domains.elastic.policy import PolicyApplication, PolicyBoundaryBusy
from areal.autopilot.revision.contracts import digest, source_digest
from areal.elastic.contracts import (
    BatchWorkload,
    ElasticObservation,
    HistoryWindow,
    Measurement,
    ResourceDecision,
    TrainingConfig,
)
from areal.elastic.control.config import (
    ConfigApplication,
    NativeConfig,
)
from areal.elastic.control.native import ControlBusy, NativeConfigExecutor
from areal.elastic.policy.basic.optimizer import BasicOptimizer
from areal.elastic.policy.basic.policy import BasicPolicy, IdleTimeoutPolicy
from areal.elastic.state.checkpoint import atomic_json
from areal.elastic.telemetry.history import RuntimeHistory

SCOPE = "native-async-v1"


class NativePolicyRuntime:
    """A fixed-layout host for policies and independently evaluated revisions.

    Pass an instance as PPOTrainer(elastic_policy=...). A measured profile must
    carry this executor's native_execution_id; old serial/lifecycle receipts
    cannot be activated here. Without a profile this records the timeout baseline.
    """

    def __init__(self, *, profile=None, settings=None, revision=None, bundle=None):
        self.profile = None if profile is None else json.loads(json.dumps(profile))
        self.settings = dataclasses.asdict(
            BasicOptimizer(**dict(settings or {"horizon": 1}))
        )
        if self.settings.get("horizon", 1) != 1:
            raise ValueError("Native fixed-layout planning requires horizon=1")
        if revision is not None and profile is None:
            raise ValueError("Native revision requires an explicit measured profile")
        self.revision, self.bundle = revision, bundle
        self.owner: Any = None
        self.policy_application = None
        self.runtime_history = None
        self.policy = None
        self.history = HistoryWindow()
        self.pending = None
        self.phases = []
        self.failed = False
        self.closed = False
        self.committed_tokens = 0
        self._decision = None

    def bind(self, owner):
        if self.owner is not None:
            raise RuntimeError("A policy runtime cannot own two coordinators")
        self.owner = owner
        if owner.latest_path is not None:
            state = owner.latest_path / "policy-state.json"
            if (
                not state.is_file()
                or json.loads(state.read_text()).get("scope") != SCOPE
            ):
                raise ValueError(
                    "Cannot attach a native policy to a checkpoint without policy state; use a new state root"
                )
        config = owner.config
        # Store hashes rather than configuration secrets. Complete resource and
        # training semantics, hardware ownership and source enter this identity.
        native_source = Path(__file__).resolve().parents[3] / "trainer/on_demand.py"
        execution = {
            "scope": SCOPE,
            "training": owner.identity,
            "rollout": digest(dataclasses.asdict(config.rollout)),
            "sampling": digest(dataclasses.asdict(config.gconfig)),
            "sglang": digest(dataclasses.asdict(config.sglang)),
            "tokenizer": {
                name: sha
                for name, sha in owner.base_manifest.items()
                if "token" in name or name in {"vocab.json", "merges.txt"}
            },
            "actor_gpus": owner.gpu_uuids,
            "rollout_gpus": list(owner.options.rollout_devices),
            "lifecycle": {
                name: value
                for name, value in dataclasses.asdict(owner.options).items()
                if name != "state_root"
            },
            "native_source": source_digest(native_source.read_text()),
            "ppo_source": source_digest(
                native_source.with_name("rl_trainer.py").read_text()
            ),
            "cpu_hooks": digest(
                {
                    name: dataclasses.asdict(getattr(config, name))
                    if dataclasses.is_dataclass(getattr(config, name))
                    else getattr(config, name)
                    for name in (
                        "saver",
                        "stats_logger",
                        "perf_tracer",
                        "evaluator",
                        "recover",
                    )
                }
            ),
            "accounting": "conditional-phases; resident-rollout; no-makespan-claim",
        }
        identity = digest(execution)
        current = (TrainingConfig(len(owner.gpu_uuids)),)
        optimizer = BasicOptimizer(**self.settings)
        if (
            optimizer.candidate_configs is not None
            and optimizer.candidate_configs != current
        ):
            raise ValueError(
                "Native planning candidates must match the fixed Actor layout"
            )
        self.settings = json.loads(
            json.dumps(
                dataclasses.asdict(
                    dataclasses.replace(optimizer, candidate_configs=current)
                )
            )
        )
        if self.profile is None:
            self.profile = {
                "execution_scope": SCOPE,
                "native_execution_id": identity,
                "native_actor_world_size": len(owner.gpu_uuids),
                "compatibility": owner.compatibility,
                "inference_gpus": len(owner.options.rollout_devices),
                "measurements": [],
            }
            self.policy = IdleTimeoutPolicy(max(owner.options.idle_timeout, 1e-9))
        else:
            if (
                self.profile.get("execution_scope") != SCOPE
                or self.profile.get("native_execution_id") != identity
                or self.profile.get("native_actor_world_size") != len(owner.gpu_uuids)
                or self.profile.get("compatibility") != owner.compatibility
                or self.profile.get("inference_gpus")
                != len(owner.options.rollout_devices)
            ):
                raise ValueError(
                    "Measured profile differs from the native execution identity"
                )
            from areal.autopilot.domains.elastic.phase import fixed_layout_simulator

            self.policy = BasicPolicy(
                fixed_layout_simulator(
                    workflow_from_profile(self.profile), self.profile
                ),
                optimizer=BasicOptimizer(**self.settings),
                fallback=IdleTimeoutPolicy(max(owner.options.idle_timeout, 1e-9)),
            )
        context = {
            "scope": SCOPE,
            "native_execution_id": identity,
            "queue_id": owner.inbox.queue.queue_id,
            "checkpoint_root": str(owner.store.root),
            "inference_gpus": len(owner.options.rollout_devices),
            "optimizer_settings": self.settings,
            "idle_seconds": owner.options.idle_timeout,
        }
        self.policy_application = PolicyApplication(
            owner.root / "policy-application",
            compatibility=owner.compatibility,
            execution_context=context,
            initial_bundle_id=digest(
                {
                    "profile": self.profile,
                    "settings": self.settings,
                    "baseline": type(self.policy).__name__,
                }
            ),
        )
        self.runtime_history = RuntimeHistory(
            owner.root / "runtime-history.json",
            context={**context, "compatibility": owner.compatibility},
        )
        self.config_application = ConfigApplication(
            NativeConfigExecutor(
                owner,
                NativeConfig(
                    identity,
                    owner.config.actor.backend,
                    owner.config.rollout.backend,
                    tuple(owner.options.actor_devices),
                    tuple(owner.options.rollout_devices),
                ),
                policy_context=self.policy_application.context,
            ),
            owner.root / "config-application",
        )
        self.runtime_history.validate_version(owner.version)
        self.recover()
        atomic_json(owner.root / "native-profile.json", self.profile)
        if self.bundle is not None:
            from areal.autopilot.domains.elastic.policy import PolicyRequest

            prepared = self.prepare_policy(Path(self.bundle))
            self.apply_policy(
                PolicyRequest(
                    "startup-" + prepared.bundle_id[:16],
                    prepared.bundle_id,
                    self.policy_generation,
                    owner.version,
                ),
                prepared,
            )

    @property
    def weights_version(self):
        return self.owner.version

    @property
    def published_version(self):
        return self.owner.inbox.status()["published_version"]

    @property
    def policy_generation(self):
        return self.policy_application.state()["generation"]

    @property
    def policy_boundary_ready(self):
        return (
            not self.failed
            and not self.closed
            and self.owner.control_snapshot().boundary_ready
        )

    def observe(self, batch=None):
        now = time.monotonic()
        progress = self.weights_version
        workload = None if batch is None else self._workload(batch)
        return ElasticObservation(
            state="ACTIVE" if self.owner.actor is not None else "SUSPENDED",
            model_version=self.weights_version,
            serving_version=self.published_version,
            training_active=self.owner.actor is not None,
            batch_ready=batch is not None,
            idle_duration=0
            if self.owner.idle_since is None
            else max(0, now - self.owner.idle_since),
            gate_available=self.policy_boundary_ready,
            pending_publication=self.published_version != self.weights_version,
            committed=self.owner.active is None
            and self.published_version == self.weights_version,
            observed_at=now,
            training_config=TrainingConfig(len(self.owner.gpu_uuids)),
            next_batch=workload,
            committed_samples=progress * self.owner.inbox.batch_size,
            committed_tokens=self.committed_tokens,
            max_train_gpus=len(self.owner.gpu_uuids),
            has_checkpoint=self.owner.latest_path is not None,
        )

    @staticmethod
    def _workload(batch):
        from areal.infra.rpc.serialization import deserialize_value

        payloads = [deserialize_value(value) for value in batch.payload["trajectories"]]
        return BatchWorkload(
            len(batch.payload["episode_ids"]),
            sum(int(value["loss_mask"].sum()) for value in payloads),
            sum(int(value["attention_mask"].sum()) for value in payloads),
        )

    def _action(self, decision, observation):
        if isinstance(decision, ResourceDecision):
            if decision.action == "reconfigure" or (
                decision.config is not None
                and decision.config != observation.training_config
            ):
                raise ValueError(
                    "Native execution currently has a fixed Actor/Rollout layout"
                )
            if decision.action == "launch" and not observation.batch_ready:
                raise ValueError("Native Actor cannot prewarm without a durable batch")
            if decision.train and not observation.batch_ready:
                raise ValueError("Training requires a durable eligible batch")
            return (
                "train"
                if decision.train
                else "release"
                if decision.action == "release"
                else "keep"
            )
        if decision not in {"train", "keep", "release"}:
            raise ValueError("Invalid native resource decision")
        return decision

    def decide(self, batch):
        if self.failed:
            raise RuntimeError("Recover pending policy activation first")
        observation = self.observe(batch)
        decision = self.policy.decide(observation)
        action = self._action(decision, observation)
        if action == "train":
            self._decision = {
                "batch_id": batch.batch_id,
                "observation": dataclasses.asdict(observation),
                "history": dataclasses.asdict(self.history.snapshot()),
                "action": dataclasses.asdict(decision)
                if dataclasses.is_dataclass(decision)
                else decision,
            }
        return action

    def poll(self):
        if self.failed:
            raise RuntimeError("Recover pending policy activation first")
        if self.revision is not None:
            self.revision.tick(self)
        if self.failed:
            raise RuntimeError("Recover pending policy activation first")

    def begin_update(self, batch, trajectories):
        if self._decision is None or self._decision["batch_id"] != batch.batch_id:
            raise RuntimeError("Native update lacks its original resource decision")
        self.pending = {
            "observation": self._decision["observation"],
            "workload": dataclasses.asdict(self._workload(batch)),
            "history": self._decision["history"],
            "decision": self._decision["action"],
            "policy_generation": self.policy_generation,
            "policy_bundle_id": self.policy_application.state()["bundle_id"],
        }
        self.pending["committed_tokens"] = (
            self.committed_tokens + self.pending["workload"]["effective_tokens"]
        )

    def record_phase(self, phase, seconds):
        measurement = Measurement(
            phase,
            len(self.owner.gpu_uuids),
            seconds,
            0 if self.pending is None else self.pending["workload"]["sequence_tokens"],
        )
        self.phases.append(dataclasses.asdict(measurement))
        if phase in {"checkpoint", "publish"} and self.pending is not None:
            atomic_json(
                self.owner.root / "policy-pending.json",
                {
                    "checkpoint_id": self.owner.latest_path.name,
                    "context": self.policy_application.context,
                    "pending": self.pending,
                    "measurements": self.phases,
                },
            )

    def checkpoint_state(self, path):
        atomic_json(
            path / "policy-state.json",
            {
                "scope": SCOPE,
                "context": self.policy_application.context,
                "pending": self.pending,
                "measurements": self.phases,
            },
        )

    def published(self):
        if self.owner.latest_path is None:
            return
        checkpoint = self.owner.latest_path
        prior = self.runtime_history.snapshot(self.weights_version)["updates"]
        if not any(row["checkpoint_id"] == checkpoint.name for row in prior):
            state = self._checkpoint_record()
            if state["context"] != self.policy_application.context:
                raise ValueError("Checkpoint policy state belongs to another execution")
            pending = state["pending"]
            if pending is None:
                raise ValueError("Committed policy checkpoint has no update identity")
            measurements = [
                {
                    "phase": row["phase"],
                    "config": {
                        "world_size": row["world_size"],
                        "profile_id": row["profile_id"],
                    },
                    "seconds": row["seconds"],
                    "workload": pending["workload"],
                }
                for row in state["measurements"]
            ]
            self.runtime_history.append(
                {
                    **pending,
                    "weights_version": self.weights_version,
                    "checkpoint_id": checkpoint.name,
                    "measurements": measurements,
                    "timing_scope": SCOPE,
                    "missing_phases": [
                        phase
                        for phase in ("checkpoint", "publish")
                        if phase not in {row["phase"] for row in measurements}
                    ],
                    "data_cursor": self.owner.consumer.cursor,
                }
            )
        self._hydrate_history()
        self.pending = None
        self.phases = []

    def _checkpoint_record(self):
        state = json.loads((self.owner.latest_path / "policy-state.json").read_text())
        path = self.owner.root / "policy-pending.json"
        if path.exists():
            pending = json.loads(path.read_text())
            if pending["checkpoint_id"] == self.owner.latest_path.name:
                state = pending
        if state["context"] != self.policy_application.context:
            raise ValueError("Checkpoint policy state belongs to another execution")
        return state

    def _hydrate_history(self):
        self.history = HistoryWindow()
        for row in self.runtime_history.snapshot(self.weights_version)["updates"]:
            self.history.add_workload(BatchWorkload(**row["workload"]))
            for measured in row["measurements"]:
                self.history.add_measurement(
                    Measurement(
                        measured["phase"],
                        measured["config"]["world_size"],
                        measured["seconds"],
                        row["workload"]["sequence_tokens"],
                    )
                )
        for boundary in self.runtime_history.snapshot(self.weights_version)[
            "boundaries"
        ]:
            for row in boundary["measurements"]:
                self.history.add_measurement(
                    Measurement(
                        row["phase"], row["config"]["world_size"], row["seconds"]
                    )
                )
        if self.owner.latest_path is not None:
            self.committed_tokens = self._checkpoint_record()["pending"][
                "committed_tokens"
            ]
        if hasattr(self.policy, "history"):
            self.policy.history = self.history

    def prepare_policy(self, bundle):
        prepared = self.policy_application.prepare(Path(bundle))
        observation = self.observe()
        shadow = HistoryWindow()
        for row in self.history.snapshot().workloads:
            shadow.add_workload(row)
        for row in self.history.snapshot().measurements:
            shadow.add_measurement(row)
        prepared.policy.history = shadow
        self._action(prepared.policy.decide(observation), observation)
        return dataclasses.replace(prepared, probed=True)

    def apply_policy(self, request, prepared):
        try:
            with self.owner.control_operation():
                return self._apply_policy(request, prepared)
        except ControlBusy as error:
            raise PolicyBoundaryBusy(str(error)) from error

    def apply_config(self, request):
        with self.owner.control_operation(blocking=True):
            self.config_application.preflight(request)
            if not self.policy_boundary_ready:
                raise PolicyBoundaryBusy(
                    "Configuration requires a committed native boundary"
                )
            return self.config_application.apply(request)

    def _apply_policy(self, request, prepared):
        if not self.policy_boundary_ready:
            raise PolicyBoundaryBusy(
                "Native policy activation requires committed, published weights and no active update"
            )
        receipt = self.policy_application.preflight(request, self.weights_version)
        if receipt is not None:
            return receipt
        try:
            receipt = self.policy_application.commit(
                request, self.weights_version, prepared, self.history
            )
            self.policy = prepared.policy
            self.policy.history = self.history
            return receipt
        except BaseException:
            self.failed = True
            self.owner.fence_admission()
            raise

    def recover(self):
        prepared = self.policy_application.recover()
        if prepared is not None:
            self.policy = prepared.policy
        self._hydrate_history()
        self.pending = None
        self.phases = []
        if self.owner.latest_path is not None:
            state = self._checkpoint_record()
            self.pending = state["pending"]
            self.phases = state["measurements"]
        self.failed = False
        if self.policy_boundary_ready:
            self.config_application.recover()
        if (
            self.published_version == self.weights_version
            and self.owner.latest_path is not None
        ):
            self.published()

    def close(self):
        if self.closed:
            return
        try:
            if self.revision is not None:
                self.revision.close()
        finally:
            self.closed = True

    def released(self, seconds, gpu_seconds):
        if (
            self.owner.active is not None
            or self.published_version != self.weights_version
        ):
            return
        rows = self.runtime_history.snapshot(self.weights_version)["updates"]
        workload = next(
            (
                row["workload"]
                for row in reversed(rows)
                if row["weights_version"] == self.weights_version
            ),
            None,
        )
        if workload is None and self.owner.latest_path is not None:
            workload = self._checkpoint_record()["pending"]["workload"]
        self.runtime_history.append_boundary(
            {
                "weights_version": self.weights_version,
                "checkpoint_id": None
                if self.owner.latest_path is None
                else self.owner.latest_path.name,
                "actor_generation": self.owner.generation,
                "actor_gpu_seconds": gpu_seconds,
                "measurements": [
                    {
                        "phase": "release",
                        "seconds": seconds,
                        "config": {"world_size": len(self.owner.gpu_uuids)},
                        "workload": workload,
                    }
                ],
            }
        )
        self._hydrate_history()
