# SPDX-License-Identifier: Apache-2.0

"""CPU-owned lifecycle around the native PPO Actor and persistent online inbox."""

from __future__ import annotations

import dataclasses
import json
import os
import threading
import time
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from copy import deepcopy
from functools import wraps
from pathlib import Path
from typing import Any

from areal.elastic.control.native import ControlBusy, ControlSnapshot
from areal.elastic.control.policy import ActorPolicyRuntime, validate_unbound_policy
from areal.elastic.control.processes import (
    OWNER_ENV,
    gpu_processes,
    owned_processes,
    resolve_gpu_uuids,
    stop_owned_processes,
)
from areal.elastic.state.batch_queue import QueuedBatch
from areal.elastic.state.checkpoint import (
    CheckpointStore,
    artifact_manifest,
    atomic_json,
    recovery_digest,
)
from areal.elastic.state.online import OnlineInbox
from areal.utils import logging
from areal.utils.atomic import json_digest

logger = logging.getLogger("RLTrainer")


def _serialized(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self._lifecycle_lock:
            return method(self, *args, **kwargs)

    return call


def validate_on_demand(config: Any) -> None:
    """Reject unsupported state/placement combinations before starting services."""
    from areal.api.alloc_mode import ModelAllocation
    from areal.utils.environ import is_single_controller

    lifecycle = config.actor_lifecycle
    actor = ModelAllocation.from_str(config.actor.backend)
    rollout = ModelAllocation.from_str(config.rollout.backend)
    if not is_single_controller() or config.scheduler.type != "local":
        raise ValueError("on_demand currently requires a local single CPU coordinator")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError(
            "Start the on_demand CPU coordinator with CUDA_VISIBLE_DEVICES empty"
        )
    if config.actor._version != "v2" or config.rollout._version != "v2":
        raise ValueError("on_demand requires native v2 Actor and Rollout controllers")
    if actor.backend not in ("fsdp", "megatron") or rollout.backend != "sglang":
        raise ValueError("on_demand currently supports FSDP/Megatron with SGLang")
    if config.rollout.agent is None or config.rollout.agent.mode != "online":
        raise ValueError("on_demand requires agent.mode='online'")
    if (
        config.critic is not None
        or config.ref is not None
        or config.teacher is not None
        or config.actor.use_lora
    ):
        raise ValueError(
            "on_demand does not yet support critic/ref/teacher or LoRA state"
        )
    if (
        config.enable_offload
        or config.actor.offload
        or config.rollout.api_url is not None
    ):
        raise ValueError(
            "on_demand requires local separated workers without offloading"
        )
    for engine in (config.actor, config.rollout):
        strategy = engine.scheduling_strategy
        if strategy is not None and strategy.type != "separation":
            raise ValueError("on_demand requires separation scheduling")
        if any(
            "CUDA_VISIBLE_DEVICES" in spec.env_vars for spec in engine.scheduling_spec
        ):
            raise ValueError(
                "Configure on_demand GPU pools instead of scheduling env CUDA_VISIBLE_DEVICES"
            )
        if len(engine.scheduling_spec) != 1:
            raise ValueError(
                "on_demand requires exactly one scheduling spec per engine"
            )
    if (
        config.actor.scheduling_spec[0].gpu != 1
        or config.rollout.scheduling_spec[0].gpu <= 0
    ):
        raise ValueError(
            "on_demand requires one GPU per Actor guard and a positive Rollout GPU request"
        )
    if (actor.parallel.world_size, rollout.parallel.world_size) != (
        len(lifecycle.actor_devices),
        len(lifecycle.rollout_devices),
    ):
        raise ValueError(
            "Each fixed GPU pool must exactly match its native parallel layout"
        )
    if (
        config.gconfig.n_samples != 1
        or config.gconfig.reward_normalization
        or config.dynamic_bs
    ):
        raise ValueError(
            "on_demand currently requires independent single-sample episodes and fixed batches"
        )
    if config.rollout.consumer_batch_size != config.train_dataset.batch_size:
        raise ValueError(
            "Rollout consumer_batch_size must equal the independent episode batch size"
        )
    for norm in (config.actor.adv_norm, config.actor.reward_norm):
        if norm is not None and (
            norm.mean_level == "group" or norm.std_level == "group"
        ):
            raise ValueError(
                "Independent single-sample episodes cannot use group normalization"
            )
    if not Path(config.actor.path).is_dir():
        raise ValueError("on_demand requires a pinned, locally materialized base model")
    if Path(config.sglang.model_path).resolve() != Path(config.actor.path).resolve():
        raise ValueError("Actor and SGLang must start from the same pinned base model")


class OnDemandActor:
    """Keep the consumer fence, serving service and data progress across GPU generations.

    The caller executes the ordinary native PPO operations between next_batch()
    and commit_and_publish(). Recovery never performs a substitute PPO update.
    """

    def __init__(
        self,
        config: Any,
        ft_spec: Any,
        actor_factory: Callable[[Any], Any],
        *,
        policy_runtime: ActorPolicyRuntime | None = None,
    ) -> None:
        validate_unbound_policy(policy_runtime)
        self.config, self.ft_spec, self.actor_factory = config, ft_spec, actor_factory
        self._lifecycle_lock = threading.RLock()
        self.policy_runtime = policy_runtime
        self._policy_bound = False
        self._train_started: float | None = None
        self.options = config.actor_lifecycle
        self.root = Path(self.options.state_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.inbox = OnlineInbox(
            self.root / "inbox",
            batch_size=config.train_dataset.batch_size,
            max_staleness=config.rollout.max_head_offpolicyness,
        )
        from areal.api.alloc_mode import ModelAllocation

        self.allocation = ModelAllocation.from_str(config.actor.backend)
        # Hash immutable model files, omitting download-cache bookkeeping.
        base = Path(config.actor.path)
        self.base_manifest = artifact_manifest(base)
        files = {
            name: digest
            for name, digest in self.base_manifest.items()
            if name.endswith((".safetensors", ".bin", ".index.json"))
            or name == "config.json"
        }
        if "config.json" not in files or not any(
            name.endswith((".safetensors", ".bin")) for name in files
        ):
            raise ValueError(
                "The pinned base model has no complete configuration/weights"
            )
        self.compatibility = {
            "native_full_state": 1,
            "backend": self.allocation.backend,
            "actor": dataclasses.asdict(config.actor),
            "base_files": files,
            "ft_spec": dataclasses.asdict(ft_spec),
            "seed": config.seed,
            "batch_size": self.inbox.batch_size,
            "max_staleness": self.inbox.max_staleness,
        }
        # Canonical JSON makes tuples/enums compare identically after recovery.
        self.compatibility = json.loads(json.dumps(self.compatibility))
        self.identity = recovery_digest(self.compatibility)
        self.store = CheckpointStore(
            self.root / "checkpoints",
            keep=self.options.checkpoint_keep,
            free_space_floor=self.options.checkpoint_free_space_floor,
        )
        self.latest_path: Path | None = None
        self.latest_manifest: dict[str, Any] | None = None
        self.actor: Any | None = None
        self.rollout: Any | None = None
        self.weight_meta: Any | None = None
        self.active: QueuedBatch | None = None
        self.owner: str | None = None
        self.ready_since: float | None = None
        self.idle_since: float | None = None
        self.started_at: float | None = None
        self.generation = 0
        self._closed = False
        self._capture_cpu: Callable[[], dict[str, Any]] = lambda: {}
        self._restore_cpu: Callable[[dict[str, Any]], None] = lambda state: None
        self.consumer = self.inbox.queue.consumer(self._read_latest)
        try:
            self.inbox.set_admission_open(False)
            self.version = (
                self.latest_manifest["global_step"] if self.latest_manifest else 0
            )
            previous = self.root / "state.json"
            if policy_runtime is None and (self.root / "policy-application").exists():
                raise ValueError(
                    "Resume this coordinator with its native policy runtime"
                )
            if previous.exists():
                journal = json.loads(previous.read_text())
                self.generation = journal["generation"]
                if journal.get("owner"):
                    raise RuntimeError(
                        "Previous coordinator left an unconfirmed Actor generation; preserve its journal for cleanup"
                    )
            self.gpu_uuids = resolve_gpu_uuids(list(self.options.actor_devices))
            rollout_gpus = resolve_gpu_uuids(list(self.options.rollout_devices))
            present = gpu_processes()
            if any(present.get(gpu) for gpu in self.gpu_uuids + rollout_gpus):
                raise RuntimeError("The explicitly assigned GPU pools are not empty")
            self._event("STOPPED")
        except BaseException:
            self.consumer.close()
            raise

    def control_snapshot(self) -> ControlSnapshot:
        with self._lifecycle_lock:
            published = self.inbox.status()["published_version"]
            ready = (
                self.owner is None
                if self.actor is None
                else self.ready_since is not None
            )
            return ControlSnapshot(
                self._closed,
                not self._closed
                and ready
                and self.active is None
                and published == self.version,
                self.generation,
                self.version,
                published,
                json_digest(
                    {
                        "checkpoint": None
                        if self.latest_path is None
                        else self.latest_path.name,
                        "cursor": None
                        if self.latest_manifest is None
                        else self.latest_manifest["data_cursor"],
                        "queue_id": self.inbox.queue.queue_id,
                    }
                ),
            )

    @contextmanager
    def control_operation(self, *, blocking: bool = False):
        if not self._lifecycle_lock.acquire(blocking=blocking):
            raise ControlBusy("Native lifecycle is busy")
        try:
            yield self.control_snapshot()
        finally:
            self._lifecycle_lock.release()

    def fence_admission(self) -> None:
        self.inbox.set_admission_open(False)

    def _read_latest(self) -> dict[str, Any] | None:
        if not (self.store.root / "latest.json").exists():
            self.latest_path, self.latest_manifest = None, None
        else:
            self.latest_path, self.latest_manifest = self.store.latest(
                self.compatibility
            )
        return self.latest_manifest

    def _event(self, state: str, **details: Any) -> None:
        event = {
            "state": state,
            "wall_time": time.time(),
            "monotonic": time.monotonic(),
            "generation": self.generation,
            "version": getattr(self, "version", 0),
            "owner": self.owner,
            **details,
        }
        with (self.root / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(event, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        atomic_json(self.root / "state.json", event)

    def bind(
        self,
        rollout: Any,
        weight_meta: Any,
        *,
        capture_cpu: Callable[[], dict[str, Any]],
        restore_cpu: Callable[[dict[str, Any]], None],
    ) -> None:
        self.rollout, self.weight_meta = rollout, weight_meta
        self._capture_cpu, self._restore_cpu = capture_cpu, restore_cpu
        if self.latest_path is not None:
            restore_cpu(json.loads((self.latest_path / "coordinator.json").read_text()))
        self._initial_cpu_state = deepcopy(capture_cpu())
        rollout._ensure_initialized()
        versions = rollout.serving_versions()
        if not versions or any(version != self.version for version in versions):
            if self.latest_path is None:
                raise RuntimeError(
                    "Initial serving weights do not match the immutable base version"
                )
            self.ensure_actor()
            self._publish()
        elif self.inbox.status()["published_version"] != self.version:
            self.inbox.confirm_publication(self.version)
        if self.policy_runtime is not None:
            self.policy_runtime.bind(self)
            self._policy_bound = True
        self.inbox.set_admission_open(True)

    def _assert_pool_free(self) -> None:
        present = gpu_processes()
        if any(present.get(gpu) for gpu in self.gpu_uuids):
            raise RuntimeError("Actor GPU pool is not confirmed empty")

    @_serialized
    def ensure_actor(self) -> Any:
        if self.actor is not None:
            return self.actor
        if self.owner is not None:
            raise RuntimeError(
                "Previous Actor generation still requires release confirmation"
            )
        self._assert_pool_free()
        self.owner = uuid.uuid4().hex
        self.generation += 1
        self.started_at = time.monotonic()
        self._event("STARTING")
        actor_config = deepcopy(self.config.actor)
        for spec in actor_config.scheduling_spec:
            spec.env_vars[OWNER_ENV] = self.owner
        self.actor = self.actor_factory(actor_config)
        self.actor.create_process_group(parallel_strategy=self.allocation.parallel)
        self.actor.initialize(
            role="actor",
            addr=None,
            ft_spec=self.ft_spec,
            wait=True,
            worker_seed=self.config.seed,
        )
        if self.latest_path is not None:
            self.actor.full_checkpoint(
                "load",
                path=str(self.latest_path),
                backend=self.allocation.backend,
                identity_sha256=self.identity,
                version=self.version,
            )
        if self.config.perf_tracer is not None:
            self.actor.config_perf_tracer(self.config.perf_tracer, role="actor")
        self.actor.connect_engine(self.rollout, self.weight_meta)
        self.ready_since = time.monotonic()
        self.idle_since = self.ready_since
        self._event("READY", startup_seconds=self.ready_since - self.started_at)
        if self._policy_bound:
            self.policy_runtime.record_phase(
                "restore" if self.latest_path else "start",
                self.ready_since - self.started_at,
            )
        return self.actor

    def next_batch(self) -> list[dict[str, Any]]:
        while True:
            with self._lifecycle_lock:
                if self._closed:
                    raise RuntimeError("On-demand coordinator is closed")
                batch = self._poll_batch()
            if batch is not None:
                return batch
            time.sleep(self.options.poll_interval)

    def _poll_batch(self) -> list[dict[str, Any]] | None:
        from areal.infra.rpc.serialization import deserialize_value

        if self.active is not None:
            raise RuntimeError("The previous native PPO update is still active")
        if self._policy_bound:
            self.policy_runtime.poll()
        if self.inbox.status()["published_version"] != self.version:
            self.ensure_actor()
            self._publish()
        identity = self.inbox.seal_batch(self.version)
        if identity is not None:
            if (
                self._policy_bound
                and self.policy_runtime.decide(self.consumer.peek()) != "train"
            ):
                return None
            self.ensure_actor()
            batch = self.consumer.claim()
            if (
                batch is None
                or batch.batch_id != identity
                or batch.policy_version != self.version
            ):
                raise RuntimeError(
                    "The reserved batch does not match restored Actor progress"
                )
            trajectories = [
                deserialize_value(value) for value in batch.payload["trajectories"]
            ]
            for trajectory in trajectories:
                minimum, maximum = self.inbox._inspect_payload(trajectory)
                if (
                    maximum > self.version
                    or self.version - minimum > self.inbox.max_staleness
                ):
                    raise RuntimeError(
                        "Reserved token versions violate the native training bound"
                    )
            if self._policy_bound:
                self.policy_runtime.begin_update(batch, trajectories)
            self.active = batch
            self._train_started = time.monotonic()
            self.idle_since = None
            self._event(
                "TRAINING",
                batch_id=identity,
                episode_ids=batch.payload["episode_ids"],
            )
            return trajectories
        self.maybe_suspend()
        return None

    @_serialized
    def commit_and_publish(self, version: int) -> None:
        if self.actor is None or self.active is None or version != self.version + 1:
            raise RuntimeError("No matching native PPO update is ready to commit")
        if self._policy_bound:
            self.policy_runtime.record_phase(
                "train", time.monotonic() - self._train_started
            )
        self.actor.set_version(version)
        self.consumer.mark_applied(self.active)
        started = time.monotonic()

        def write(path: Path) -> None:
            self.actor.full_checkpoint(
                "save",
                path=str(path),
                backend=self.allocation.backend,
                identity_sha256=self.identity,
                version=version,
            )
            atomic_json(path / "coordinator.json", self._capture_cpu())
            if self._policy_bound:
                self.policy_runtime.checkpoint_state(path)

        generation = self.store.commit(
            step=version,
            world_size=self.allocation.parallel.world_size,
            compatibility=self.compatibility,
            write=write,
            estimated_bytes=self.options.checkpoint_estimated_bytes,
            data_cursor=self.consumer.cursor,
        )
        manifest = self._read_latest()
        if self.latest_path != generation or manifest is None:
            raise RuntimeError("Checkpoint publication requires reconciliation")
        self.consumer.acknowledge(manifest)
        self.version = version
        self.active = None
        self._event(
            "COMMITTED",
            checkpoint=generation.name,
            checkpoint_seconds=time.monotonic() - started,
        )
        if self._policy_bound:
            self.policy_runtime.record_phase("checkpoint", time.monotonic() - started)
        self._publish()

    def _publish(self) -> None:
        if self.actor is None or self.rollout is None:
            raise RuntimeError("Publication requires live native Actor and Rollout")
        started = time.monotonic()
        self._event("PUBLISHING")
        self.actor.update_weights(self.weight_meta.with_version(self.version))
        versions = self.rollout.serving_versions()
        if not versions or any(version != self.version for version in versions):
            raise RuntimeError(
                "Serving replicas have not all acknowledged committed weights"
            )
        self.inbox.confirm_publication(self.version)
        if self._policy_bound:
            self.policy_runtime.record_phase("publish", time.monotonic() - started)
            self.policy_runtime.published()
        self.inbox.set_admission_open(True)
        self.idle_since = time.monotonic()
        self._event(
            "READY",
            publication_seconds=self.idle_since - started,
            serving_versions=versions,
        )

    @_serialized
    def maybe_suspend(self, now: float | None = None) -> bool:
        if self.actor is None or self.active is not None:
            return False
        now = time.monotonic() if now is None else now
        if self.inbox.status()["published_version"] != self.version:
            return False
        if self.inbox.seal_batch(self.version) is not None:
            self.idle_since = None
            return False
        if self.idle_since is None:
            self.idle_since = now
        if (
            now - self.idle_since < self.options.idle_timeout
            or now - self.ready_since < self.options.min_ready_seconds
        ):
            return False
        if self._policy_bound and self.policy_runtime.decide(None) != "release":
            return False
        self._release()
        return True

    def _release(self) -> None:
        if self.actor is None and self.owner is None:
            return
        release_started = time.monotonic()
        self._cleanup_event("RELEASING")
        if self.actor is not None:
            # No resource creator may outlive the ownership probe below.
            self.actor.quiesce_initialization()
            if self.ready_since is not None:
                try:
                    self.actor.save_perf_tracer(force=True)
                except Exception:
                    logger.warning(
                        "Actor tracer flush failed during release", exc_info=True
                    )
            try:
                self.actor.destroy()
            except Exception:
                # A dead Actor cannot acknowledge AWEX teardown. Only after
                # proving its entire generation has stopped may the surviving
                # inference peers be disconnected without training-side RPCs.
                if self.owner is None:
                    raise
                stop_owned_processes(self.owner)
                self._confirm_stopped()
                self.actor.destroy(training_stopped=True)
        if self.owner is not None:
            stop_owned_processes(self.owner)
        self._confirm_stopped()
        gpu_seconds = (time.monotonic() - self.started_at) * len(self.gpu_uuids)
        self.actor, self.owner = None, None
        self.ready_since, self.idle_since, self.started_at = None, None, None
        self._cleanup_event("STOPPED", actor_gpu_seconds=gpu_seconds)
        if self._policy_bound:
            self.policy_runtime.released(
                time.monotonic() - release_started, gpu_seconds
            )

    def _confirm_stopped(self) -> None:
        if self.owner is not None and owned_processes(self.owner):
            raise RuntimeError("Actor descendants remain alive")
        self._assert_pool_free()

    def _cleanup_event(self, state: str, **details: Any) -> None:
        try:
            self._event(state, **details)
        except Exception:
            logger.error("Cannot persist lifecycle cleanup evidence", exc_info=True)

    @_serialized
    def recover_uncertain(self, error: Exception) -> None:
        """Stop the failed generation, then let latest decide whether an update committed."""
        try:
            self.inbox.set_admission_open(False)
        except Exception:
            logger.error("Cannot persist the recovery admission fence", exc_info=True)
        self._cleanup_event("RECOVERING", error=str(error))
        self._release()
        manifest = self._read_latest()
        self.consumer.restore_checkpoint(manifest)
        self.version = manifest["global_step"] if manifest else 0
        self.active = None
        if self.latest_path is not None:
            self._restore_cpu(
                json.loads((self.latest_path / "coordinator.json").read_text())
            )
        else:
            self._restore_cpu(deepcopy(self._initial_cpu_state))
        if self._policy_bound:
            self.policy_runtime.recover()
        if self.inbox.status()["published_version"] != self.version or any(
            version != self.version for version in self.rollout.serving_versions()
        ):
            self.ensure_actor()
            self._publish()
        self.inbox.set_admission_open(True)

    @_serialized
    def close(self) -> None:
        if self._closed:
            return
        errors = []
        try:
            self.inbox.set_admission_open(False)
        except Exception as error:
            errors.append(error)
        try:
            self._release()
        except Exception as error:
            if self.actor is not None or self.owner is not None:
                raise ExceptionGroup(
                    "Actor release incomplete; retain consumer ownership",
                    [*errors, error],
                ) from error
            # Measurement persistence can fail after release was proved. It must
            # not keep the queue fence or revision worker alive.
            errors.append(error)
        try:
            if self.policy_runtime is not None and getattr(
                self.policy_runtime, "owner", self
            ) in (None, self):
                self.policy_runtime.close()
        except Exception as error:
            errors.append(error)
        finally:
            self.consumer.close()
            self._closed = True
        if errors:
            raise ExceptionGroup("Coordinator cleanup incomplete", errors)
