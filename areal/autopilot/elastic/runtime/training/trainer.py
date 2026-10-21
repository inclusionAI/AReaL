# SPDX-License-Identifier: Apache-2.0

"""Bind one optimizer update per queued batch to a recoverable checkpoint."""

import time
from collections.abc import Callable
from typing import Any

from areal.autopilot.contracts import TrainingConfig
from areal.autopilot.elastic.runtime.state.batch_queue import BatchQueue, QueuedBatch


class QueuedTrainer:
    """Single caller; keep the consumer lock until the GPU group has stopped.

    Update failures have unknown outcomes. Never retry on the same live actor:
    stop it and reopen from the published checkpoint instead. Before the first
    checkpoint, recovery requires the same immutable base model and seeded init.
    """

    def __init__(self, group: Any, queue: BatchQueue):
        self.group = group
        self.queue = queue
        self.consumer = None
        self._failed = False
        self.latest_checkpoint = None
        self.committed_step = 0
        self.committed_cursor = None
        self.committed_samples = self.committed_tokens = 0
        self._samples = self._tokens = 0
        self.last_reconfiguration_error = None

    @property
    def version(self) -> int:
        return self.group.step

    @property
    def is_active(self) -> bool:
        return self.group.actor is not None

    @property
    def is_committed(self) -> bool:
        """Whether execution has a known outcome bound to the committed cursor."""
        return (
            self.consumer is not None
            and not self._failed
            and self.group.step == self.committed_step
            and self.consumer.cursor == self.committed_cursor
        )

    def has_pending_batch(self) -> bool:
        """Observe queue readiness without leasing or advancing its cursor."""
        if self.consumer is None:
            raise RuntimeError("Queued trainer is not prepared")
        return self.consumer.peek() is not None

    @property
    def training_config(self):
        return getattr(self.group, "training_config", None)

    @property
    def max_train_gpus(self):
        return len(getattr(self.group, "pool_devices", ()))

    def next_workload(self):
        if self.consumer is None or not hasattr(self.group, "workload"):
            return None
        batch = self.consumer.peek()
        return None if batch is None else self.group.workload(batch.payload)

    def validate_config(self, config: TrainingConfig) -> None:
        """Reject unsupported targets before changing admission or live workers."""
        validator = getattr(self.group, "validate_config", None)
        if validator is not None:
            validator(config)
        elif config.profile_id:
            raise ValueError("Training executor does not validate named actor profiles")

    def configure(self, config: TrainingConfig):
        """Resize only at a committed boundary, preserving the consumer fence."""
        if not self.is_committed:
            raise RuntimeError("Cannot reconfigure uncommitted training")
        self.validate_config(config)
        previous = self.training_config
        self.last_reconfiguration_error = None
        measurements = []
        if config == previous:
            return measurements
        if self.is_active:
            started = time.monotonic()
            self.suspend()
            measurements.append(("release", previous, time.monotonic() - started))
        try:
            self.group.configure(config)
            if self.latest_checkpoint is not None:
                started = time.monotonic()
                self.group.start(restore=True)
                measurements.append(("restore", config, time.monotonic() - started))
            if self.group.step != self.committed_step:
                raise RuntimeError("Target did not restore the committed step")
        except Exception as error:
            self._failed = True
            # Never launch a third group while target cleanup is uncertain.
            self.group.stop()
            self.group.configure(previous)
            if self.latest_checkpoint is not None:
                started = time.monotonic()
                self.group.start(restore=True)
                measurements.append(("restore", previous, time.monotonic() - started))
            if self.group.step != self.committed_step:
                raise RuntimeError(
                    "Rollback did not restore the committed step"
                ) from error
            self._failed = False
            self.last_reconfiguration_error = f"{type(error).__name__}: {error}"
        return measurements

    def next_batch_id(self) -> str | None:
        """Validate the next trajectory on CPU before allocating any GPU."""
        if self.consumer is None:
            raise RuntimeError("Queued trainer is not prepared")
        batch = self.consumer.peek()
        if batch is None:
            return None
        self._validate_trajectory(batch)
        return batch.batch_id

    def _validate_trajectory(self, batch: QueuedBatch) -> None:
        if batch.policy_version != self.version:
            raise ValueError("Queue head has a stale policy version")
        if batch.payload["policy_version"] != batch.policy_version:
            raise ValueError("Queue envelope and trajectory versions disagree")
        self.group.validate_trajectory(batch.payload)

    def prepare(self) -> None:
        """Fence consumption and recover the cursor without allocating any GPU."""
        if self.consumer is not None or self.group.actor is not None:
            raise RuntimeError("Queued trainer already exists")
        manifest = None

        def read_checkpoint():
            nonlocal manifest
            if (self.group.store.root / "latest.json").exists():
                self.latest_checkpoint, manifest = self.group.store.latest(
                    self.group.compatibility
                )
            return manifest

        self.latest_checkpoint = None
        self.consumer = self.queue.consumer(read_checkpoint)
        self.group.step = manifest["global_step"] if manifest is not None else 0
        self.committed_step = self.group.step
        self.committed_cursor = self.consumer.cursor
        try:
            self._samples = self._tokens = 0
            if hasattr(self.group, "workload"):
                for payload in self.consumer.applied_payloads():
                    workload = self.group.workload(payload)
                    self._samples += workload.samples
                    self._tokens += workload.effective_tokens
            self.committed_samples, self.committed_tokens = self._samples, self._tokens
            if (
                manifest is not None
                and "data_progress" in manifest
                and manifest["data_progress"]
                != {"samples": self._samples, "effective_tokens": self._tokens}
            ):
                raise ValueError(
                    "Checkpoint logical data progress disagrees with queue prefix"
                )
        except BaseException:
            self.consumer.close()
            self.consumer = None
            self._failed = True
            raise
        self._failed = False

    def activate(self) -> None:
        """Start or restore workers while retaining the prepared consumer fence."""
        if self.consumer is None or self.group.actor is not None:
            raise RuntimeError("Expected a prepared suspended trainer")
        if self._failed:
            raise RuntimeError("Training outcome uncertain; stop and recover first")
        try:
            self.group.start(restore=self.latest_checkpoint is not None)
            if self.group.step != self.consumer.cursor["applied_batches"]:
                raise RuntimeError("Training step disagrees with queue cursor")
        except BaseException:
            self._failed = True
            self.stop()
            raise

    def start(self) -> None:
        self.prepare()
        self.activate()

    def _require_active(self) -> None:
        if self.consumer is None or self.group.actor is None:
            raise RuntimeError("Queued trainer is not active")
        if self._failed:
            raise RuntimeError("Training outcome uncertain; stop and recover first")

    def train_next(
        self, update: Callable[[QueuedBatch], dict[str, Any]] | None = None
    ) -> dict[str, Any] | None:
        """Apply one trajectory, or an explicit fixture callback receiving its envelope."""
        self._require_active()
        try:
            batch = self.consumer.claim()
            if batch is None:
                return None
            before = self.group.step
            if update is None:
                self._validate_trajectory(batch)
                result = self.group.train_trajectory(batch.payload)
            else:
                result = update(batch)
            if self.group.step != before + 1:
                raise RuntimeError(
                    "Each queue batch must complete one optimizer update"
                )
            self.consumer.mark_applied(batch)
            if hasattr(self.group, "workload"):
                workload = self.group.workload(batch.payload)
                self._samples += workload.samples
                self._tokens += workload.effective_tokens
            return {**result, "batch_id": batch.batch_id, "batch_digest": batch.digest}
        except BaseException:
            self._failed = True
            raise

    def checkpoint(self):
        self._require_active()
        try:
            self.group.data_progress = {
                "samples": self._samples,
                "effective_tokens": self._tokens,
            }
            path = self.group.checkpoint(data_cursor=self.consumer.cursor)
            published, manifest = self.group.store.latest(self.group.compatibility)
            if path != published:
                raise RuntimeError("Checkpoint is not the published generation")
            self.consumer.acknowledge(manifest)
            self.latest_checkpoint = path
            self.committed_step = self.group.step
            self.committed_cursor = manifest["data_cursor"]
            self.committed_samples, self.committed_tokens = self._samples, self._tokens
            return path
        except BaseException:
            self._failed = True
            raise

    def suspend(self) -> None:
        """Release only checkpointed workers; keep queue ownership while idle."""
        self._require_active()
        if (
            self.group.step != self.committed_step
            or self.consumer.cursor != self.committed_cursor
        ):
            raise RuntimeError("Checkpoint uncommitted updates before suspension")
        try:
            self.group.stop()
        except BaseException:
            self._failed = True
            raise

    def stop(self) -> None:
        """Discard uncommitted updates; explicitly checkpoint before a clean pause."""
        self._failed = True
        # If teardown fails, retain the lock: another consumer must not run while
        # an old worker may still own an in-flight update.
        self.group.stop()
        if self.consumer is not None:
            self.consumer.close()
            self.consumer = None
