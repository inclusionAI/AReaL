# SPDX-License-Identifier: Apache-2.0

"""Revision and accepted activation for native PPO.

This adapter never creates workers or executes PPO. Native async measurements are
conditional service times; they do not establish makespan or GPU savings.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from areal.autopilot.domains.elastic.policy import PolicyApplication, PolicyBoundaryBusy
from areal.autopilot.revision.contracts import digest
from areal.elastic.contracts import (
    HistoryWindow,
)
from areal.elastic.control.boundary import ControlBusy
from areal.elastic.control.config_application import (
    ConfigApplication,
    NativeConfig,
    NativeConfigExecutor,
)
from areal.elastic.monitor.collector import SCOPE
from areal.elastic.policy.adapter import PolicyAdapter


class RevisionAdapter(PolicyAdapter):
    """Extend ordinary native policies with independently evaluated revisions.

    Pass an instance as PPOTrainer(elastic_policy=...). A measured profile must
    carry this executor's native_execution_id; old serial/lifecycle receipts
    cannot be activated here. The AReaL Monitor supplies observations and history.
    """

    supports_policy_activation = True

    def __init__(self, *, profile=None, settings=None, revision=None, bundle=None):
        super().__init__(profile=profile, settings=settings)
        if revision is not None and profile is None:
            raise ValueError("Native revision requires an explicit measured profile")
        self.revision, self.bundle = revision, bundle
        self.hooks = self
        self.policy_application = None
        self.failed = False

    def bind(self, owner):
        super().bind(owner)
        identity = self.execution_id
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
        self.recover()
        if self.bundle is not None:
            from areal.autopilot.domains.elastic.policy import PolicyRequest

            prepared = self.prepare_policy(Path(self.bundle))
            self.apply_policy(
                PolicyRequest(
                    "startup-" + prepared.bundle_id[:16],
                    prepared.bundle_id,
                    self.policy_generation,
                    owner.applied_version,
                ),
                prepared,
            )

    @property
    def runtime_history(self):
        """Revision windows come from the same default Monitor as ordinary policies."""
        return self.owner.monitor.runtime_history

    @property
    def measurement_ready(self):
        return self.owner.monitor.metrics_available

    def measurement_failed(self, error):
        self.owner.monitor.mark_unavailable(error)

    def policy_metadata(self) -> dict:
        return {
            **super().policy_metadata(),
            "policy_generation": self.policy_generation,
            "policy_bundle_id": self.policy_application.state()["bundle_id"],
        }

    @property
    def weights_version(self):
        """Revision host protocol: applied weights, independent of checkpoint saving."""
        return self.owner.applied_version

    @property
    def policy_generation(self):
        return self.policy_application.state()["generation"]

    @property
    def policy_boundary_ready(self):
        return (
            not self.failed
            and not self.closed
            and self.measurement_ready
            and self.owner.control_snapshot().boundary_ready
        )

    def observe(self, batch=None, *, now=None):
        observation = super().observe(batch, now=now)
        return dataclasses.replace(
            observation,
            gate_available=not self.failed and observation.gate_available,
        )

    def _require_decision_boundary(self):
        super()._require_decision_boundary()
        if self.failed:
            raise RuntimeError("Recover pending policy activation first")

    def poll(self):
        if self.failed:
            raise RuntimeError("Recover pending policy activation first")
        if self.revision is not None and self.measurement_ready:
            self.revision.tick(self)
        if self.failed:
            raise RuntimeError("Recover pending policy activation first")

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
                    "Configuration requires a completed, published native boundary"
                )
            return self.config_application.apply(request)

    def _apply_policy(self, request, prepared):
        if not self.policy_boundary_ready:
            raise PolicyBoundaryBusy(
                "Native policy activation requires applied, published weights and no active update"
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
        if hasattr(self.policy, "history"):
            self.policy.history = self.history
        self.failed = False
        if self.policy_boundary_ready:
            self.config_application.recover()

    def close(self):
        if self.closed:
            return
        try:
            if self.revision is not None:
                self.revision.close()
        finally:
            super().close()
