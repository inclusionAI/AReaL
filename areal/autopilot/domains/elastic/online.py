# SPDX-License-Identifier: Apache-2.0

"""Bounded lifecycle revisions from frozen committed runtime windows.

The model gate scores measured service phases, not end-to-end wall time. The
optimizer gate uses that frozen executable model. Training quality is not scored.
"""

from __future__ import annotations

import fcntl
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Protocol

from areal.autopilot.domains.elastic.contracts import (
    BASELINE_SOURCE,
    history_from_dict,
    observation_from_dict,
)
from areal.autopilot.domains.elastic.cycle import _fixed_json, _stage
from areal.autopilot.domains.elastic.evaluator import RevisionEvaluator
from areal.autopilot.domains.elastic.policy import (
    PolicyBoundaryBusy,
    PolicyRequest,
    export_accepted_policy,
)
from areal.autopilot.revision.contracts import digest
from areal.autopilot.revision.loop import RevisionLoop
from areal.elastic.policy.basic.optimizer import BasicOptimizer
from areal.utils.atomic import atomic_json


def runtime_problems(profile, window, *, settings=None, runner=None):
    """Split whole updates before exposing calibration labels to an agent."""
    updates = window["updates"]
    if len(updates) < 4:
        raise ValueError("Revision needs at least four measured committed updates")
    split = max(2, min(len(updates) - 1, int(len(updates) * 0.6)))
    calibration, validation = updates[:split], updates[split:]
    public_profile = {**profile, "runtime_calibration": calibration}
    cases, optimizer_cases = [], []
    for update in validation:
        observation = observation_from_dict(update["observation"])
        optimizer_cases.append(
            {"observation": asdict(observation), "history": update["history"]}
        )
        boundary_measurements = [
            row
            for boundary in window.get("boundaries", ())
            if boundary["weights_version"] == update["weights_version"]
            for row in boundary["measurements"]
        ]
        for row in [*update["measurements"], *boundary_measurements]:
            if row["workload"] is None or row["config"] is None:
                continue
            from areal.elastic.contracts import BatchWorkload, TrainingConfig

            observed = replace(
                observation,
                training_config=TrainingConfig(**row["config"]),
                next_batch=BatchWorkload(**row["workload"]),
            )
            # Only calibration history enters the phase estimate: validation timing
            # labels stay in the parent. This is a conditional phase-time check.
            prior = history_from_dict(calibration[-1]["history"])
            cases.append(
                {
                    "phase": row["phase"],
                    "observation": asdict(observed),
                    "history": asdict(prior),
                    "forecast": [{"workload": row["workload"], "wait_seconds": 0}],
                    "plan": [{"config": row["config"], "release_before": False}],
                    "observed": {
                        "feasible": True,
                        "seconds": row["seconds"],
                        "total_gpu_seconds": row["seconds"]
                        * (
                            observed.training_config.world_size
                            + profile.get("inference_gpus", 1)
                        ),
                    },
                }
            )
    if not cases:
        raise ValueError(
            "Runtime window contains no measured phases with known workload"
        )
    return RevisionEvaluator(
        "simulator", public_profile, cases, settings=settings, runner=runner
    ), optimizer_cases


class LifecycleRevisionCycle:
    """Model acceptance → frozen model → optimizer revalidation/revision → bundle."""

    def __init__(
        self,
        model_evaluator,
        optimizer_cases,
        root: Path,
        *,
        initial_model_source=None,
        initial_optimizer_source=None,
        parent_bundle: Path | None = None,
        idle_seconds=30,
    ):
        if model_evaluator.target != "simulator" or not model_evaluator.phase_only:
            raise ValueError(
                "Runtime fixture revisions require a lifecycle phase model"
            )
        self.model, self.cases, self.root = model_evaluator, optimizer_cases, root
        root.mkdir(parents=True, exist_ok=True)
        self.model_source = initial_model_source or BASELINE_SOURCE["simulator"]
        self.optimizer_source = initial_optimizer_source or BASELINE_SOURCE["optimizer"]
        self.parent, self.idle_seconds = parent_bundle, idle_seconds

    def run(
        self,
        model_agent,
        optimizer_agent,
        *,
        max_model_rounds=2,
        max_optimizer_rounds=2,
    ):
        with (self.root / "cycle.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return self._run(
                model_agent,
                optimizer_agent,
                max_model_rounds=max_model_rounds,
                max_optimizer_rounds=max_optimizer_rounds,
            )

    def _run(
        self, model_agent, optimizer_agent, *, max_model_rounds, max_optimizer_rounds
    ):
        if any(
            type(n) is not int or n < 1
            for n in (max_model_rounds, max_optimizer_rounds)
        ):
            raise ValueError("Revision round limits must be positive")
        _fixed_json(
            self.root / "context.json",
            {
                "model": self.model.benchmark_sha256,
                "optimizer_cases": digest(self.cases),
                "model_source": digest(self.model_source),
                "optimizer_source": digest(self.optimizer_source),
                "rounds": [max_model_rounds, max_optimizer_rounds],
            },
        )
        model = _stage(
            RevisionLoop(
                model_agent,
                self.model,
                self.root / "model",
                initial_source=self.model_source,
            ),
            max_model_rounds,
        )
        selected_source = model.source.read_text()
        model_report = json.loads((model.root / model.record["report"]).read_text())
        phase_reference = {
            "cases": json.loads(self.model._public_json),
            "prediction_sha256": model_report["metrics"]["prediction_sha256"],
        }
        search = RevisionEvaluator(
            "optimizer",
            self.model.calibration,
            self.cases,
            settings=self.model.settings,
            runner=self.model.runner,
            reference_simulator_source=selected_source,
            phase_reference=phase_reference,
        )
        _fixed_json(
            self.root / "handoff.json",
            {
                "model_source_sha256": model.record["source_sha256"],
                "optimizer_benchmark": search.benchmark_sha256,
            },
        )
        optimizer = _stage(
            RevisionLoop(
                optimizer_agent,
                search,
                self.root / "optimizer",
                initial_source=self.optimizer_source,
            ),
            max_optimizer_rounds,
        )
        bundle = None
        if optimizer.record["accepted"]:
            bundle = export_accepted_policy(
                self.root / "optimizer",
                search,
                self.root / "bundles",
                model_revision_root=self.root / "model"
                if model.record["accepted"]
                else None,
                model_verification_root=self.root / "model"
                if not model.record["accepted"]
                else None,
                parent_bundle=self.parent,
                idle_seconds=self.idle_seconds,
            )
        elif model.record["accepted"]:
            bundle = export_accepted_policy(
                self.root / "model",
                self.model,
                self.root / "bundles",
                optimizer_verification_root=self.root / "optimizer",
                parent_bundle=self.parent,
                idle_seconds=self.idle_seconds,
            )
        result = {
            "status": "complete",
            "model": model.record,
            "optimizer": optimizer.record,
            "bundle": None if bundle is None else str(bundle.resolve()),
        }
        _fixed_json(self.root / "result.json", result)
        return result


class RevisionHost(Protocol):
    """Committed policy boundary; independent of worker/trainer object graphs."""

    policy: Any
    policy_application: Any
    runtime_history: Any
    weights_version: int
    published_version: int
    policy_generation: int
    policy_boundary_ready: bool

    def prepare_policy(self, bundle: Path): ...
    def apply_policy(self, request: PolicyRequest, prepared): ...


class RuntimeRevisionLoop:
    """One background CPU revision; apply only at a current committed boundary.

    Started windows and round limits are durable budgets. An interrupted window is
    resumed in its original directory, rather than receiving a fresh proposal cap.
    """

    def __init__(
        self,
        root: Path,
        *,
        profile: dict,
        model_agent,
        optimizer_agent,
        settings: dict,
        every_updates=4,
        max_windows=3,
        max_model_rounds=2,
        max_optimizer_rounds=2,
        runner=None,
        idle_seconds=30,
        scope="lifecycle",
        horizon=2,
    ):
        if any(
            type(n) is not int or n < 1
            for n in (
                every_updates,
                max_windows,
                max_model_rounds,
                max_optimizer_rounds,
            )
        ):
            raise ValueError("Runtime revision budgets must be positive")
        self.root = root
        self.profile = json.loads(json.dumps(profile, allow_nan=False))
        self.scope, self.horizon = scope, horizon
        if scope == "serial-joint-timing-v1":
            from areal.autopilot.domains.elastic.serial import joint_settings

            self.settings = json.loads(json.dumps(asdict(joint_settings(settings))))
        elif scope in {"lifecycle", "native-async-v1"}:
            self.settings = asdict(BasicOptimizer(**settings))
        else:
            raise ValueError("Unknown runtime revision scope")
        root.mkdir(parents=True, exist_ok=True)
        self.agents = model_agent, optimizer_agent
        self.every, self.maximum = every_updates, max_windows
        self.rounds = max_model_rounds, max_optimizer_rounds
        self.runner, self.idle_seconds = runner, idle_seconds
        self.future, self.pending = None, None
        self._closing = False
        self._lock = (root / "runtime.lock").open("a")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            _fixed_json(
                root / "context.json",
                {
                    "profile": digest(self.profile),
                    "settings": self.settings,
                    "every_updates": every_updates,
                    "max_windows": max_windows,
                    "rounds": list(self.rounds),
                    "idle_seconds": idle_seconds,
                    **(
                        {"scope": scope, "horizon": horizon}
                        if scope != "lifecycle"
                        else {}
                    ),
                },
            )
        except BaseException:
            self._lock.close()
            raise
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="policy-revision"
        )

    def _run(self, directory):
        request = json.loads((directory / "request.json").read_text())
        if self.scope == "serial-joint-timing-v1":
            return self._run_joint(directory, request)
        model, cases = runtime_problems(
            self.profile, request["window"], settings=self.settings, runner=self.runner
        )
        parent = request["parent_bundle"]
        sources = (
            BASELINE_SOURCE
            if parent is None
            else {
                name: json.loads(Path(parent).read_text())["components"][name]["source"]
                for name in ("simulator", "optimizer")
            }
        )
        result = LifecycleRevisionCycle(
            model,
            cases,
            directory / "cycle",
            initial_model_source=sources["simulator"],
            initial_optimizer_source=sources["optimizer"],
            parent_bundle=None if parent is None else Path(parent),
            idle_seconds=self.idle_seconds,
        ).run(
            *self.agents,
            max_model_rounds=self.rounds[0],
            max_optimizer_rounds=self.rounds[1],
        )
        return directory, request, result

    def _run_joint(self, directory, request):
        from areal.autopilot.domains.elastic.serial import (
            SerialRevisionCycle,
            runtime_joint_problems,
        )

        model, cases = runtime_joint_problems(
            self.profile, request["window"], settings=self.settings, runner=self.runner
        )
        parent = request["parent_bundle"]
        sources = (
            BASELINE_SOURCE
            if parent is None
            else {
                name: json.loads(Path(parent).read_text())["components"][name]["source"]
                for name in ("serial_model", "joint_optimizer")
            }
        )
        result = SerialRevisionCycle(
            model,
            cases,
            directory / "cycle",
            initial_model_source=sources["serial_model"],
            initial_optimizer_source=sources["joint_optimizer"],
            parent_bundle=None if parent is None else Path(parent),
            idle_seconds=self.idle_seconds,
            horizon=self.horizon,
        ).run(
            *self.agents,
            max_model_rounds=self.rounds[0],
            max_optimizer_rounds=self.rounds[1],
        )
        return directory, request, result

    def _check_initial_joint(self, host):
        from areal.elastic.policy.basic.optimizer import JointOptimizer
        from areal.elastic.policy.basic.serial_planning import (
            OnlineJointPolicy,
            serial_model_from_profile,
        )

        policy = host.policy
        model = serial_model_from_profile(self.profile, idle_seconds=self.idle_seconds)
        if (
            type(policy) is not OnlineJointPolicy
            or type(policy.optimizer) is not JointOptimizer
            or digest(asdict(policy.optimizer)) != digest(self.settings)
            or policy.horizon != self.horizon
            or type(policy.simulator) is not type(model)
            or policy.simulator.timing_predictor is not None
            or policy.simulator.phases != model.phases
            or policy.simulator.episodes != model.episodes
            or policy.simulator.transitions != model.transitions
            or policy.simulator.idle_seconds != self.idle_seconds
        ):
            raise ValueError("Initial joint policy differs from the revision baseline")

    def tick(self, host: RevisionHost):
        if self._lock.closed:
            raise RuntimeError("Runtime revision loop is closed")
        if not getattr(host, "measurement_ready", True):
            return
        if host.policy_application is None or host.runtime_history is None:
            raise ValueError(
                "Runtime revisions require policy activation and committed history stores"
            )
        if (
            host.policy_application.execution_context.get("scope", "lifecycle")
            != self.scope
        ):
            raise ValueError("Runtime revision and policy application scopes differ")
        if self.profile.get("compatibility") != host.policy_application.compatibility:
            raise ValueError(
                "Revision profiles do not match the live training contract"
            )
        if host.policy_generation == 0 and self.scope == "serial-joint-timing-v1":
            self._check_initial_joint(host)
        elif host.policy_generation == 0:
            from areal.elastic.policy.basic.policy import BasicPolicy
            from areal.elastic.policy.basic.profile import workflow_from_profile
            from areal.elastic.policy.basic.simulator import WorkflowSimulator

            policy = host.policy
            if (
                type(policy) is not BasicPolicy
                or not isinstance(policy.simulator, WorkflowSimulator)
                or policy.simulator.workflow != workflow_from_profile(self.profile)
                or type(policy.optimizer) is not BasicOptimizer
                or asdict(policy.optimizer) != self.settings
                or policy.fallback.idle_seconds != self.idle_seconds
            ):
                raise ValueError(
                    "Initial live policy differs from the revision baseline"
                )
        execution = {
            "policy_context": host.policy_application.context,
            "history_context": host.runtime_history.context,
        }
        binding = self.root / "execution.json"
        if binding.exists():
            prior = json.loads(binding.read_text())
            if (
                prior.get("policy_context") == execution["policy_context"]
                and prior.get("history_context") != execution["history_context"]
                and hasattr(host, "measurement_failed")
            ):
                host.measurement_failed(
                    ValueError(
                        "Revision history series changed; use a fresh revision root"
                    )
                )
                return
        _fixed_json(binding, execution)
        if self.future is not None and self.future.done():
            try:
                self.pending = self.future.result()
            except Exception as error:
                directory = self._running
                atomic_json(
                    directory / "outcome.json",
                    {"status": "failed", "error": f"{type(error).__name__}: {error}"},
                )
            self.future = None
        if self.pending is not None:
            directory, request, result = self.pending
            if result["bundle"] is None:
                atomic_json(directory / "outcome.json", {"status": "unchanged"})
                self.pending = None
            else:
                application = host.policy_application
                state = application.state()
                if state["generation"] != request["policy_generation"]:
                    # A prior apply may have committed immediately before a crash.
                    old = state["receipts"].get(directory.name)
                    atomic_json(
                        directory / "outcome.json",
                        {"status": "applied" if old else "stale", "receipt": old},
                    )
                    self.pending = None
                elif host.policy_boundary_ready:
                    selected = None
                    try:
                        prepared = host.prepare_policy(Path(result["bundle"]))
                        selected = PolicyRequest(
                            directory.name,
                            prepared.bundle_id,
                            state["generation"],
                            host.weights_version,
                        )
                        receipt = host.apply_policy(selected, prepared)
                    except PolicyBoundaryBusy:
                        return  # Episode/publication boundary is busy; retry later.
                    except (ValueError, TypeError, RuntimeError) as error:
                        if (
                            selected is not None
                            and host.weights_version
                            != selected.expected_weights_version
                        ):
                            return
                        atomic_json(
                            directory / "outcome.json",
                            {"status": "activation_rejected", "error": str(error)},
                        )
                    else:
                        atomic_json(
                            directory / "outcome.json",
                            {"status": "applied", "receipt": receipt},
                        )
                    self.pending = None
        if (
            self._closing
            or self.future is not None
            or self.pending is not None
            or not host.policy_boundary_ready
        ):
            return
        windows = sorted(self.root.glob("window-*"))
        unfinished = [d for d in windows if not (d / "outcome.json").exists()]
        if unfinished:
            directory = unfinished[0]
        else:
            last = max(
                [
                    json.loads((d / "request.json").read_text())["window"][
                        "through_version"
                    ]
                    for d in windows
                ],
                default=0,
            )
            if len(windows) >= self.maximum or host.weights_version < max(
                4, last + self.every
            ):
                return
            try:
                window = host.runtime_history.snapshot(host.published_version)
            except (OSError, ValueError, KeyError, TypeError) as error:
                if not hasattr(host, "measurement_failed"):
                    raise
                host.measurement_failed(error)
                return
            if len(window["updates"]) < 4:
                return
            directory = self.root / f"window-{len(windows) + 1:04d}"
            directory.mkdir()
            state = host.policy_application.state()
            parent = (
                None
                if state["generation"] == 0
                else str(
                    host.policy_application.root
                    / "bundles"
                    / state["bundle_id"]
                    / "bundle.json"
                )
            )
            atomic_json(
                directory / "request.json",
                {
                    "policy_generation": state["generation"],
                    "parent_bundle": parent,
                    "window": window,
                },
            )
        self._running = directory
        self.future = self.executor.submit(self._run, directory)

    def close(self, host=None):
        self._closing = True
        try:
            self.executor.shutdown(wait=True)
            if host is not None:
                # Finish under the same ownership lock; no new window can start.
                self.tick(host)
        finally:
            self._lock.close()
