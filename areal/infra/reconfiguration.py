# SPDX-License-Identifier: Apache-2.0
"""Snapshot primitives for explicit runtime reconfiguration.

This first layer saves and restores *registered components*, not a complete
configuration transition. The caller must establish a quiescent trainer boundary
and register every required state owner (including scheduler/RNG state where the
backend does not save it). The trainer channel drains registered rollout work
and registers FSDP/Megatron scheduler and per-rank RNG state. Restoration offers
a weight-synchronization callback before loading owners, then restores RNGs last.
TrainerSnapshotRestore wires these stages into explicit PPOTrainer startup.
ConfigurationTransition executes mappings approved by the external Validator.
Process restart stays in the harness. A successful component restore alone
does not authorize a config transition.

Use one coordinator per snapshot. Checkpoint callbacks must dispatch any required
collectives to ALL participating ranks; this module never runs rank-local save
calls on behalf of a distributed controller. Snapshots must be on shared storage
visible to those ranks. After publication they must remain immutable.

Only load trusted snapshots: registered Python state is serialized with pickle,
just as dataloader state is in the existing recovery implementation. Checksums
detect corruption, not an untrusted snapshot author's intent.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import inspect
import json
import math
import os
import pickle
import random
import re
import stat
import sys
import threading
import uuid
from collections import deque
from collections.abc import Callable, Mapping
from concurrent.futures import Future
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

_SCHEMA_VERSION = 1
_SCOPE = "registered_components"
_MANIFEST = "manifest.json"
_STATE = "state.pkl"
CONFIGURATION_APPLY_ADAPTER = "areal-v1-drained-restart-v1"
FSDP_LAYOUT_RNG_POLICY = "fsdp-mesh-reseed-v1"
MEGATRON_LAYOUT_RNG_POLICY = "megatron-model-parallel-reseed-v1"
MEGATRON_PORTABLE_FORMAT = "areal-megatron-fully-reshardable-v1"
_PARALLEL_DIMS = (
    "data_parallel_size",
    "tensor_parallel_size",
    "pipeline_parallel_size",
    "context_parallel_size",
    "expert_parallel_size",
    "expert_tensor_parallel_size",
)


class SnapshotError(RuntimeError):
    """Snapshot I/O, integrity or participant failure; never a fresh-start signal."""


class Stateful(Protocol):
    """An explicitly registered owner of host-side or supplemental training state."""

    def state_dict(self) -> Any: ...

    def load_state_dict(self, state: Any) -> None: ...


def capture_rng_state(*, include_device: bool = True) -> dict[str, Any]:
    """Capture global RNGs; custom generators must be registered separately."""
    import numpy as np
    import torch

    state = {
        "python": random.getstate(),
        "numpy": copy.deepcopy(np.random.get_state()),
        "torch": torch.get_rng_state().cpu().clone(),
        "device_type": None,
        "device": None,
    }
    if include_device:
        from areal.infra.platforms import current_platform

        if current_platform.device_type != "cpu":
            state["device_type"] = current_platform.device_type
            state["device"] = current_platform.get_rng_state().cpu().clone()
            dtensor_rng = sys.modules.get("torch.distributed.tensor._random")
            tracker = getattr(dtensor_rng, "_rng_tracker", None)
            state["dtensor"] = {
                "initialized": tracker is not None,
                "distribute_region_enabled": (
                    tracker.distribute_region_enabled if tracker is not None else None
                ),
            }
    seeding = sys.modules.get("areal.utils.seeding")
    if seeding is not None:
        shuffler = seeding._SHUFFLER
        state["seeding"] = {
            "seed": seeding._SEED,
            "base_seed": seeding._BASE_SEED,
            "shuffler": None if shuffler is None else vars(shuffler).copy(),
        }
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore the assigned device only; do not initialize other accelerators."""
    if state.get("policy") == FSDP_LAYOUT_RNG_POLICY:
        _install_fsdp_rng(state)
        return
    import numpy as np
    import torch

    if state["device_type"] is not None:
        from areal.infra.platforms import current_platform

        if state["device_type"] != current_platform.device_type:
            raise SnapshotError("Runtime RNG device type does not match")
        if "dtensor" in state and "_dtensor_mesh" in state:
            _restore_dtensor_metadata(state["dtensor"], state["_dtensor_mesh"])
        current_platform.set_rng_state(state["device"])
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "seeding" in state:
        from areal.utils import seeding

        metadata = state["seeding"]
        seeding._SEED = metadata["seed"]
        seeding._BASE_SEED = metadata["base_seed"]
        seeding._SHUFFLER = None
        if metadata["shuffler"] is not None:
            seeding._SHUFFLER = seeding.Shuffler()
            vars(seeding._SHUFFLER).update(metadata["shuffler"])


def _restore_dtensor_metadata(metadata: dict, mesh: Any) -> None:
    from torch.distributed.tensor import _random as dtensor_rng

    if not metadata["initialized"]:
        dtensor_rng._rng_tracker = None
        return
    if dtensor_rng._rng_tracker is None:
        # Without this, the first random DTensor op in a fresh worker would
        # broadcast WORLD rank zero's state over the just-restored rank streams.
        dtensor_rng._rng_tracker = dtensor_rng.OffsetBasedRNGTracker(
            mesh, run_state_sync=False
        )
    dtensor_rng._rng_tracker.distribute_region_enabled = metadata[
        "distribute_region_enabled"
    ]


def _install_fsdp_rng(state: Mapping[str, Any]) -> None:
    """Install new logical streams only after all checkpoint/owner loads."""
    import numpy as np
    import torch
    from torch.distributed.tensor._random import manual_seed

    from areal.infra.platforms import current_platform
    from areal.utils import seeding

    if state["device_type"] != current_platform.device_type:
        raise SnapshotError("Runtime RNG device type does not match")
    # The pinned torch 2.9/2.10 API also initializes the DTensor tracker without
    # a WORLD-rank-zero broadcast. A plain torch.manual_seed omits that step.
    manual_seed(state["tensor_seed"], state["mesh"])
    random.seed(state["host_seed"])
    np.random.seed(state["host_seed"])
    cpu_rng = torch.Generator(device="cpu").manual_seed(state["host_seed"])
    torch.set_rng_state(cpu_rng.get_state())
    seeding._SEED = state["tensor_seed"]
    seeding._BASE_SEED = state["tensor_seed"]
    seeding._SHUFFLER = None


def _derived_seed(material: dict, domain: str, coordinates: list[int]) -> int:
    digest = hashlib.sha256(_json_bytes([material, domain, coordinates])).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1) + 1


def _encode_runtime_state(state: Any) -> str:
    # Ordinary RPCs reinterpret tensors as batch shards/remote references and
    # convert tuples to lists. An opaque string preserves RNG state exactly.
    return base64.b64encode(pickle.dumps(state, protocol=5)).decode("ascii")


def _decode_runtime_state(state: str) -> Any:
    if not isinstance(state, str):
        raise SnapshotError("Runtime state must be an encoded string")
    return pickle.loads(base64.b64decode(state, validate=True))


def collect_runtime_state(
    engine: Any,
    *,
    identity: Callable[[], dict[str, Any]],
    capture_rng: Callable[[], Any],
) -> str:
    """Collect one independent packet per rank over the engine's CPU group.

    Every rank must enter this call once, while quiescent. All DP heads return
    the same complete packet, so the controller may select any head's result.
    Capture/serialization errors still participate in the collective.
    """
    import torch.distributed as dist

    group = engine.cpu_group
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    try:
        if (engine.rank, engine.world_size) != (rank, world_size):
            raise SnapshotError("Engine rank mapping does not match its CPU group")
        local = {
            "rank": rank,
            "identity": identity(),
            "scheduler": engine.lr_scheduler.state_dict(),
            "lrs": [group["lr"] for group in engine.optimizer.param_groups],
            "rng": capture_rng(),
        }
        packet = {"state": _encode_runtime_state(local)}
    except Exception as exc:
        packet = {"error": f"rank {rank}: {type(exc).__name__}: {exc}"}
    gathered = [None] * world_size
    dist.all_gather_object(gathered, packet, group=group)
    errors = [item["error"] for item in gathered if "error" in item]
    if errors:
        raise SnapshotError("Runtime state capture failed: " + "; ".join(errors))
    states = [_decode_runtime_state(item["state"]) for item in gathered]
    if any(
        item["rank"] != index or item["identity"] != states[0]["identity"]
        for index, item in enumerate(states)
    ):
        raise SnapshotError("Runtime rank identities are inconsistent")
    # Reuse the original encoded packets: re-pickling deserialized torch
    # storages can embed process-local identifiers and differ across heads.
    return _encode_runtime_state(
        {"version": 1, "ranks": [item["state"] for item in gathered]}
    )


def select_runtime_state(
    engine: Any, state: str, *, identity: dict[str, Any]
) -> dict[str, Any]:
    """Select the old rank exactly, or execute an explicit layout mapping."""
    decoded = _decode_runtime_state(state)
    if decoded.get("version") == 2:
        return _select_mapped_runtime_state(engine, decoded, identity)
    ranks = [_decode_runtime_state(item) for item in decoded["ranks"]]
    if (
        decoded["version"] != 1
        or len(ranks) != engine.world_size
        or not 0 <= engine.rank < len(ranks)
        or any(
            item["rank"] != index or item["identity"] != identity
            for index, item in enumerate(ranks)
        )
    ):
        raise SnapshotError("Runtime topology or scheduler definition does not match")
    local = ranks[engine.rank]
    if identity.get("backend") == "fsdp" and "dtensor" in local["rng"]:
        # DeviceMesh is process-local; never serialize it into a snapshot/RPC.
        local["rng"]["_dtensor_mesh"] = engine.parallel_helper.world_mesh["tp"]
    return local


def load_runtime_scheduler(
    engine: Any,
    state: str,
    *,
    identity: dict[str, Any],
    reset_steps: bool = False,
) -> None:
    """Restore scheduler progress after DCP, without taking an extra LR step."""
    local = select_runtime_state(engine, state, identity=identity)
    groups = engine.optimizer.param_groups
    mapped_megatron = local.get("layout_mapped") and identity["backend"] == "megatron"
    if not mapped_megatron and len(groups) != len(local["lrs"]):
        raise SnapshotError("Runtime optimizer parameter groups do not match")
    if reset_steps:
        # Megatron's load_state_dict adds the saved count to num_steps. Native
        # DCP may already have loaded it; make this explicit restore absolute.
        engine.lr_scheduler.num_steps = 0
    engine.lr_scheduler.load_state_dict(local["scheduler"])
    if mapped_megatron:
        # Target stages/expert ownership may change group order/count. Native
        # OptimizerParamScheduler.load_state_dict applies the saved absolute
        # step to target groups using their lr_mult/wd_mult; never zip old LRs.
        return
    # LambdaLR.load_state_dict does not update optimizer LRs. Restore these
    # explicitly; Megatron's scheduler may have updated them while loading.
    for group, lr in zip(groups, local["lrs"], strict=True):
        group["lr"] = lr
    if local.get("layout_mapped") and engine.optimizer_config.type == "adam":
        # Native DCP restores source param_groups, including this execution
        # option. VLM+TP requires the target's non-fused AdamW implementation.
        fused = not (engine.is_vision_model and engine.parallel_helper.tp_enabled)
        for group in groups:
            group["fused"] = fused


def _select_mapped_runtime_state(engine: Any, packet: dict, identity: dict) -> dict:
    mapping = TrainingLayoutTransition(**packet["mapping"])
    source = _decode_runtime_state(packet["source"])
    ranks = [_decode_runtime_state(item) for item in source["ranks"]]

    def parallelism(dims):
        value = dict(zip(_PARALLEL_DIMS, dims))
        if mapping.backend == "megatron":
            value.update(
                virtual_pipeline_parallel_size=mapping.virtual_pipeline_size,
                use_sequence_parallel=dims[1] > 1,
            )
        return value

    source_identity = dict(identity, parallelism=parallelism(mapping.source))
    if (
        source.get("version") != 1
        or identity["backend"] != mapping.backend
        or identity["parallelism"] != parallelism(mapping.target)
        or len(ranks) != mapping.source_world_size
        or engine.world_size != mapping.target_world_size
        or not 0 <= engine.rank < engine.world_size
        or any(
            item["rank"] != i or item["identity"] != source_identity
            for i, item in enumerate(ranks)
        )
    ):
        raise SnapshotError(
            "Runtime identity does not match the approved layout mapping"
        )
    common = ranks[0]
    if any(
        _json_bytes(item["scheduler"]) != _json_bytes(common["scheduler"])
        or (
            mapping.backend == "fsdp"
            and _json_bytes(item["lrs"]) != _json_bytes(common["lrs"])
        )
        for item in ranks[1:]
    ):
        raise SnapshotError(
            "Source ranks disagree on scheduler progress or learning rates"
        )
    material = packet["seed_material"]
    if material["policy"] != mapping.rng_policy:
        raise SnapshotError("Runtime RNG policy does not match transition")
    if mapping.backend == "megatron":
        return dict(
            common,
            rank=engine.rank,
            identity=identity,
            layout_mapped=True,
            rng={
                "policy": mapping.rng_policy,
                "device_type": identity["device_type"],
                "root_seed": _derived_seed(material, "model-parallel", []),
            },
        )
    helper = engine.parallel_helper
    world_mesh = helper.world_mesh
    # The native FSDP path uses TP for random DTensor operations. Logical
    # (DP, SP) replicas get separate streams; TP peers share a tensor seed.
    mesh = world_mesh["tp"]
    coordinates = [world_mesh["dp"].get_local_rank(), world_mesh["sp"].get_local_rank()]
    return dict(
        common,
        rank=engine.rank,
        identity=identity,
        layout_mapped=True,
        rng={
            "policy": mapping.rng_policy,
            "device_type": identity["device_type"],
            "mesh": mesh,
            "host_seed": _derived_seed(material, "host", [engine.rank]),
            "tensor_seed": _derived_seed(material, "tensor", coordinates),
        },
    )


def install_megatron_rng(engine: Any, state: Mapping[str, Any]) -> None:
    """Install new streams with the pinned MCore TP/EP/ETP rank semantics."""
    import numpy as np
    import torch
    from megatron.core import mpu, tensor_parallel

    from areal.infra.platforms import current_platform

    if (
        state["policy"] != MEGATRON_LAYOUT_RNG_POLICY
        or state["device_type"] != current_platform.device_type
    ):
        raise SnapshotError("Megatron RNG policy/device type does not match")
    # Match native model-parallel seeding: pipeline stages differ, DP and CP
    # peers share the base. Native code adds TP and expert rank offsets.
    seed = state["root_seed"] + 100 * mpu.get_pipeline_model_parallel_rank()
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.set_rng_state(torch.Generator(device="cpu").manual_seed(seed).get_state())
    tensor_parallel.model_parallel_cuda_manual_seed(seed)
    engine.seed = seed


class EngineRuntimeState:
    """Supplement native model/optimizer DCP with scheduler and per-rank RNGs."""

    def __init__(self, engine: Any):
        for name in (
            "runtime_state_dict",
            "load_runtime_state_dict",
            "restore_runtime_rng",
        ):
            callback = getattr(engine, name, None)
            if not callable(callback) or inspect.iscoroutinefunction(callback):
                raise TypeError(f"Engine must provide blocking {name}()")
        self.engine = engine
        self._pending = None

    def state_dict(self) -> str:
        return self.engine.runtime_state_dict()

    def load_state_dict(self, state: str) -> None:
        _call(self.engine.load_runtime_state_dict, state)
        self._pending = state

    def finalize_restore(self) -> None:
        if self._pending is None:
            raise SnapshotError("Engine RNG restore was not staged")
        _call(self.engine.restore_runtime_rng, self._pending)
        self._pending = None


class ProcessRngState:
    """Host global RNGs, finalized after rollout and all other state loads.

    The controller process does not own a training accelerator. Register this
    after engine runtime owners so its RNGs are finalized after their RPCs.
    """

    def __init__(self):
        self._pending = None

    def state_dict(self) -> dict[str, Any]:
        return capture_rng_state(include_device=False)

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._pending = copy.deepcopy(state)

    def finalize_restore(self) -> None:
        if self._pending is None:
            raise SnapshotError("Host RNG restore was not staged")
        restore_rng_state(self._pending)
        self._pending = None


class RolloutStateful(Stateful, Protocol):
    """Rollout state owner with a submission fence independent of pause/resume."""

    def freeze_and_drain(self, timeout: float) -> None: ...

    def release_snapshot(self) -> None: ...

    def get_version(self) -> int: ...


class RolloutInputBuffer:
    """Checkpointable replacement for a generator holding half a loader batch.

    Owns the loader cursor together with its already-fetched remainder. Restore
    before creating an iterator: set the saved sampler epoch first, then load
    the loader's state. No generator, iterator or worker connection is pickled.
    """

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.epoch = 0
        self._remaining = deque()
        self._iterator = None
        self._restored = False

    def __next__(self):
        exhausted = False
        while not self._remaining:
            if self._iterator is None:
                if not self._restored:
                    self._set_epoch()
                self._iterator = iter(self.dataloader)
                self._restored = False
            try:
                self._remaining.extend(next(self._iterator))
            except StopIteration:
                if exhausted:
                    raise RuntimeError("Rollout dataloader yielded no inputs") from None
                exhausted = True
                self.epoch += 1
                self._iterator = None
        return self._remaining.popleft()

    def _set_epoch(self):
        sampler = getattr(self.dataloader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(self.epoch)

    def state_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "remaining": list(self._remaining),
            "dataloader": self.dataloader.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if self._iterator is not None or self._remaining:
            raise RuntimeError("Restore requires a fresh rollout input buffer")
        if type(state["epoch"]) is not int or state["epoch"] < 0:
            raise ValueError("Invalid dataloader epoch")
        self.epoch = state["epoch"]
        self._set_epoch()
        self.dataloader.load_state_dict(state["dataloader"])
        self._remaining = deque(copy.deepcopy(state["remaining"]))
        self._restored = True


@dataclass(frozen=True)
class SnapshotContext:
    """Source configuration and actual boundary supplied by the trusted caller.

    ``step_info`` contains epoch, epoch_step, global_step and steps_per_epoch,
    describing the LAST completed training step. Policy version is recorded
    independently; it is not inferred from the step counter. Configuration
    admissibility and source-to-target state mappings belong to the Validator.
    """

    run_id: str
    runtime_revision: str
    configuration_id: str
    state_id: str
    step_info: dict[str, int]
    policy_version: int


@dataclass(frozen=True)
class SnapshotRef:
    """Explicit snapshot location and manifest digest; no 'latest' discovery."""

    path: str
    snapshot_id: str
    manifest_sha256: str


@dataclass(frozen=True)
class TrainingLayoutTransition:
    """Native DCP resharding plus deterministic new logical RNG streams.

    Dimensions are DP, TP, PP, CP, EP, ETP. Model/optimizer tensors are restored
    by native DCP, never by copying one source rank's tensor state to new ranks.
    """

    source: tuple[int, int, int, int, int, int]
    target: tuple[int, int, int, int, int, int]
    rng_policy: str = FSDP_LAYOUT_RNG_POLICY
    backend: str = "fsdp"
    virtual_pipeline_size: int = 1

    def __post_init__(self):
        for name in ("source", "target"):
            dims = tuple(getattr(self, name))
            if len(dims) != 6 or any(type(v) is not int or v < 1 for v in dims):
                raise ValueError("Invalid training layout dimensions")
            if self.backend == "fsdp" and (dims[2] != 1 or dims[4:] != (1, 1)):
                raise ValueError("Native FSDP layout mapping supports DP/TP/CP only")
            if math.prod(dims[:4]) % (dims[2] * dims[4] * dims[5]):
                raise ValueError("Expert dimensions must divide the training world")
            object.__setattr__(self, name, dims)
        policy = {
            "fsdp": FSDP_LAYOUT_RNG_POLICY,
            "megatron": MEGATRON_LAYOUT_RNG_POLICY,
        }.get(self.backend)
        if policy is None or self.rng_policy != policy or self.source == self.target:
            raise ValueError("Invalid training layout RNG mapping")
        if (
            type(self.virtual_pipeline_size) is not int
            or self.virtual_pipeline_size < 1
        ):
            raise ValueError("Invalid virtual pipeline size")
        if self.virtual_pipeline_size > 1 and (
            self.backend != "megatron"
            or any(dims[2] == 1 for dims in (self.source, self.target))
        ):
            raise ValueError("Virtual pipeline requires pipeline parallelism")

    @property
    def source_world_size(self) -> int:
        return math.prod(self.source[:4])

    @property
    def target_world_size(self) -> int:
        return math.prod(self.target[:4])


@dataclass(frozen=True)
class ConfigurationTransition:
    """A prevalidated mapping for a drained, end-of-step restart.

    The caller binds the resolved source/target configurations and Validator
    result in plan_id. Scheduler definition, sampling and staleness semantics
    stay fixed. Inference instances are rebuilt by normal initialization.
    Training RNGs survive unchanged layouts; an explicit training_layout uses
    native checkpoint resharding and records deterministic new RNG streams.
    """

    plan_id: str
    source_configuration_id: str
    target_configuration_id: str
    source_limits: tuple[int, int, int]
    target_limits: tuple[int, int, int]
    adapter_id: str = CONFIGURATION_APPLY_ADAPTER
    training_layout: TrainingLayoutTransition | None = None

    def __post_init__(self):
        if self.adapter_id != CONFIGURATION_APPLY_ADAPTER:
            raise ValueError("Unknown configuration transition adapter")
        for name in ("plan_id", "source_configuration_id", "target_configuration_id"):
            if not re.fullmatch(r"sha256:[0-9a-f]{64}", getattr(self, name)):
                raise ValueError(f"Invalid transition {name}")
        for name in ("source_limits", "target_limits"):
            value = tuple(getattr(self, name))
            if len(value) != 3 or any(
                type(v) is not int or v < (0 if i == 2 else 1)
                for i, v in enumerate(value)
            ):
                raise ValueError("Invalid transition rollout limits")
            object.__setattr__(self, name, value)
        if self.source_limits[1:] != self.target_limits[1:]:
            raise ValueError("Batch size and staleness have no state mapping")
        if isinstance(self.training_layout, dict):
            object.__setattr__(
                self,
                "training_layout",
                TrainingLayoutTransition(**self.training_layout),
            )
        if self.training_layout is not None and not isinstance(
            self.training_layout, TrainingLayoutTransition
        ):
            raise TypeError("training_layout must be a TrainingLayoutTransition")

    def map_training_state(self, state: str, snapshot: SnapshotRef) -> str:
        """Keep source identity and bind reseeding to a recorded plan/snapshot.

        The same plan and snapshot always produce the same target streams;
        no clock, PID, host placement or Python hash() participates.
        """
        if self.training_layout is None:
            return state
        return _encode_runtime_state(
            {
                "version": 2,
                "source": state,
                "mapping": asdict(self.training_layout),
                "seed_material": {
                    "policy": self.training_layout.rng_policy,
                    "plan_id": self.plan_id,
                    "snapshot_id": snapshot.snapshot_id,
                    "manifest_sha256": snapshot.manifest_sha256,
                    "target_configuration_id": self.target_configuration_id,
                    "target_layout": self.training_layout.target,
                    "role": "actor",
                },
            }
        )

    def map_rollout_state(self, state: dict, target_limits: list[int]) -> dict:
        """Verify snapshot/target binding, then copy only the effective limit.

        These are execution integrity checks, not a new configuration decision.
        Any mismatch means the captured state or runtime violated the plan.
        """
        staleness = state["staleness"]
        if (
            staleness["schema_version"] != 1
            or tuple(staleness["limits"]) != self.source_limits
            or tuple(target_limits) != self.target_limits
            or staleness["counters"]["running"] != 0
        ):
            raise SnapshotError("Rollout snapshot/target does not match transition")
        mapped = copy.deepcopy(state)
        mapped["staleness"]["limits"] = list(self.target_limits)
        return mapped


@dataclass(frozen=True)
class SnapshotRestoreRequest:
    """Explicit source for a fresh trainer; no automatic 'latest' fallback.

    Ordinary recovery uses the same configuration. A cross-configuration caller
    supplies a transition approved before the source was paused.
    """

    snapshot: SnapshotRef
    expected_context: SnapshotContext
    transition: ConfigurationTransition | None = None


@dataclass(frozen=True)
class SnapshotStopResult:
    """Committed boundary returned when train() stops for reconfiguration.

    Rollout admission remains closed. Leave the trainer context (or call
    trainer.close()) to clean up workers. This is not a process-exit receipt.
    """

    snapshot: SnapshotRef
    context: SnapshotContext


@dataclass(frozen=True)
class RestoreResult:
    """Registered state was loaded; runtime confirmation is still required."""

    snapshot: SnapshotRef
    context: SnapshotContext
    checkpoint_roles: tuple[str, ...]
    state_components: tuple[str, ...]


@dataclass(frozen=True)
class CheckpointParticipant:
    """Blocking callbacks for one role's model and optimizer checkpoint.

    ``save`` may start asynchronous I/O, but ``wait_saved`` MUST wait for all
    ranks, finalize metadata and propagate errors before returning. Even after
    ``save`` raises, ``wait_saved`` is attempted to settle any started writes.
    All three callbacks return None; backend-specific timeouts belong inside
    them. An explicit no-op wait is appropriate only for a synchronous saver.

    ``format_id`` identifies the checkpoint encoding/backend compatibility, not
    the parallel layout. Layout migration must be implemented and validated
    separately; matching format IDs alone does not establish transition support.
    """

    format_id: str
    save: Callable[[Path], None]
    wait_saved: Callable[[], None]
    load: Callable[[Path], None]

    @classmethod
    def for_engine(
        cls,
        engine: Any,
        *,
        format_id: str,
        wait_saved: Callable[[], None] | None = None,
    ) -> CheckpointParticipant:
        """Adapt existing AReaL engine/controller save/load with optimizer state.

        Completion defaults to the engine/controller's wait_for_checkpoint().
        An explicit callback may supply another proven all-rank operation,
        for example a controller wait with a custom timeout. There is no
        implicit no-op fallback for engines without a completion interface.
        This adapter does not restore versions or synchronize inference weights;
        the caller must register/synchronize those states before resuming.
        """
        for name in ("save", "load"):
            if not callable(getattr(engine, name, None)):
                raise TypeError(f"Engine must provide {name}()")
        if wait_saved is None:
            wait_saved = getattr(engine, "wait_for_checkpoint", None)
        if not callable(wait_saved):
            raise TypeError(
                "Engine must provide wait_for_checkpoint() or an explicit wait_saved"
            )

        weight_format = "dcp"
        if format_id in {"areal-megatron-dcp", MEGATRON_PORTABLE_FORMAT}:
            format_id = MEGATRON_PORTABLE_FORMAT
            weight_format = "dcp-reconfiguration"

        def save(path: Path) -> None:
            from areal.api.io_struct import SaveLoadMeta

            _call(engine.save, SaveLoadMeta(str(path), weight_format, with_optim=True))

        def load(path: Path) -> None:
            from areal.api.io_struct import SaveLoadMeta

            _call(engine.load, SaveLoadMeta(str(path), weight_format, with_optim=True))

        return cls(format_id, save, wait_saved, load)


def _call(function: Callable[..., None], *args: Any) -> None:
    result = function(*args)
    if result is not None:
        if inspect.iscoroutine(result):
            result.close()
        raise TypeError("Snapshot callbacks must block and return None")


def _names(mapping: Mapping[str, Any]) -> None:
    for name in mapping:
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError(f"Invalid snapshot component name: {name!r}")


def _context_dict(context: SnapshotContext) -> dict[str, Any]:
    value = asdict(context)
    for key in ("run_id", "runtime_revision", "configuration_id", "state_id"):
        if not isinstance(value[key], str) or not value[key].strip():
            raise ValueError(f"Snapshot context requires {key}")
    step = value["step_info"]
    keys = {"epoch", "epoch_step", "global_step", "steps_per_epoch"}
    if not isinstance(step, dict) or set(step) != keys:
        raise ValueError("step_info must describe the last completed StepInfo")
    if any(type(v) is not int or v < 0 for v in step.values()):
        raise ValueError("StepInfo fields must be nonnegative integers")
    if step["steps_per_epoch"] == 0 or step["epoch_step"] >= step["steps_per_epoch"]:
        raise ValueError("epoch_step must be within steps_per_epoch")
    if type(value["policy_version"]) is not int or value["policy_version"] < 0:
        raise ValueError("policy_version must be a nonnegative integer")
    return value


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, allow_nan=False).encode("utf-8")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Duplicate snapshot manifest key: {key}")
        value[key] = item
    return value


def _write_file(path: Path, data: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _write_state(path: Path, state: dict[str, Any]) -> None:
    with path.open("xb") as stream:
        pickle.dump(state, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        os.fsync(stream.fileno())


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _inventory(root: Path, *, sync: bool = False) -> dict[str, dict[str, Any]]:
    """Stream file hashes; reject links and avoid loading model shards into RAM."""
    files = {}
    directories = [root]
    for path in sorted(root.rglob("*")):
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode):
            directories.append(path)
        elif not stat.S_ISREG(mode):
            raise SnapshotError(f"Snapshot contains a link or special file: {path}")
        elif path.relative_to(root).as_posix() != _MANIFEST:
            digest = hashlib.sha256()
            size = 0
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
                    size += len(chunk)
                if sync:
                    os.fsync(stream.fileno())
            files[path.relative_to(root).as_posix()] = {
                "size": size,
                "sha256": digest.hexdigest(),
            }
    if sync:
        for directory in sorted(directories, key=lambda p: len(p.parts), reverse=True):
            _sync_directory(directory)
    return files


def _check_file_layout(files: Mapping[str, Any], roles: Mapping[str, Any]) -> None:
    if _STATE not in files:
        raise SnapshotError("Snapshot has no registered state payload")
    for role in roles:
        if not any(name.startswith(f"checkpoints/{role}/") for name in files):
            raise SnapshotError(f"Snapshot has no checkpoint files for {role}")
    for name in files:
        if name == _STATE:
            continue
        parts = Path(name).parts
        if len(parts) < 3 or parts[0] != "checkpoints" or parts[1] not in roles:
            raise SnapshotError(f"Unexpected snapshot file: {name}")


def inspect_snapshot(snapshot: SnapshotRef) -> dict[str, Any]:
    """Verify the published snapshot without loading Python or model state."""
    try:
        root = Path(snapshot.path)
        if not root.is_absolute() or root.is_symlink() or not root.is_dir():
            raise ValueError("SnapshotRef must identify an absolute snapshot directory")
        manifest_path = root / _MANIFEST
        if manifest_path.is_symlink():
            raise ValueError("Snapshot manifest must not be a symlink")
        encoded = manifest_path.read_bytes()
        if hashlib.sha256(encoded).hexdigest() != snapshot.manifest_sha256:
            raise ValueError("Snapshot manifest digest mismatch")
        manifest = json.loads(encoded, object_pairs_hook=_unique_object)
        required = {
            "schema_version",
            "scope",
            "snapshot_id",
            "context",
            "checkpoints",
            "state_components",
            "files",
        }
        if not isinstance(manifest, dict) or set(manifest) != required:
            raise ValueError("Invalid snapshot manifest fields")
        if (
            type(manifest["schema_version"]) is not int
            or manifest["schema_version"] != _SCHEMA_VERSION
        ):
            raise ValueError("Unsupported snapshot schema version")
        if (
            manifest["scope"] != _SCOPE
            or manifest["snapshot_id"] != snapshot.snapshot_id
        ):
            raise ValueError("Snapshot identity or scope mismatch")
        _context_dict(SnapshotContext(**manifest["context"]))
        roles = manifest["checkpoints"]
        if not isinstance(roles, dict) or not roles:
            raise ValueError("Snapshot has no checkpoint roles")
        _names(roles)
        if any(not isinstance(fmt, str) or not fmt.strip() for fmt in roles.values()):
            raise ValueError("Checkpoint format IDs must be nonempty strings")
        names = manifest["state_components"]
        if not isinstance(names, list) or any(not isinstance(n, str) for n in names):
            raise ValueError("Invalid state component names")
        if len(names) != len(set(names)):
            raise ValueError("Duplicate state component names")
        _names(dict.fromkeys(names))
        files = _inventory(root)
        if files != manifest["files"] or _STATE not in files:
            raise ValueError("Snapshot file inventory or checksum mismatch")
        _check_file_layout(files, roles)
        return manifest
    except SnapshotError:
        raise
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise SnapshotError(
            f"Invalid or incomplete snapshot at {snapshot.path}"
        ) from exc


class ReconfigurationManager:
    """Coordinate independent snapshots of an explicitly registered runtime.

    Existing RecoverHandler modes, frequency counters and directories are not
    modified. Register data/saver/evaluator/stats/frequency state as appropriate,
    plus any scheduler/RNG/version state omitted by the chosen backend. The
    manifest lists actual coverage rather than claiming full runtime recovery.

    A failed restore may have partially mutated participants. Discard/recreate
    them before retrying; this manager refuses further operations after a load
    failure. It never rolls back a live trainer or restarts from step zero.
    """

    def __init__(
        self,
        *,
        checkpoints: Mapping[str, CheckpointParticipant],
        statefuls: Mapping[str, Stateful],
    ):
        _names(checkpoints)
        _names(statefuls)
        if not checkpoints:
            raise ValueError("At least one checkpoint role is required")
        for role, participant in checkpoints.items():
            if not isinstance(participant, CheckpointParticipant):
                raise TypeError(f"{role} requires a CheckpointParticipant")
            if (
                not isinstance(participant.format_id, str)
                or not participant.format_id.strip()
            ):
                raise ValueError(f"{role} requires a checkpoint format ID")
            for operation in ("save", "wait_saved", "load"):
                callback = getattr(participant, operation)
                if not callable(callback) or inspect.iscoroutinefunction(callback):
                    raise TypeError(f"{role}.{operation} must be a blocking callback")
        for name, owner in statefuls.items():
            for operation in ("state_dict", "load_state_dict"):
                callback = getattr(owner, operation, None)
                if not callable(callback) or inspect.iscoroutinefunction(callback):
                    raise TypeError(f"{name} must provide blocking {operation}()")
            if hasattr(owner, "finalize_restore"):
                callback = owner.finalize_restore
                if not callable(callback) or inspect.iscoroutinefunction(callback):
                    raise TypeError(f"{name} must provide blocking finalize_restore()")
        self._checkpoints = dict(checkpoints)
        self._statefuls = dict(statefuls)
        self._lock = threading.Lock()
        self._restore_failed = False

    def _save_checkpoints(self, root: Path) -> None:
        (root / "checkpoints").mkdir()
        for role, participant in self._checkpoints.items():
            path = root / "checkpoints" / role
            path.mkdir()
            try:
                _call(participant.save, path)
            finally:
                _call(participant.wait_saved)

    def _save_state(self, root: Path) -> None:
        state = {name: owner.state_dict() for name, owner in self._statefuls.items()}
        _write_state(root / _STATE, state)

    def _publish(self, root: Path, context: SnapshotContext) -> SnapshotRef:
        context_value = _context_dict(context)
        files = _inventory(root, sync=True)
        _check_file_layout(files, self._checkpoints)
        manifest = {
            "schema_version": _SCHEMA_VERSION,
            "scope": _SCOPE,
            "snapshot_id": uuid.uuid4().hex,
            "context": context_value,
            "checkpoints": {k: v.format_id for k, v in self._checkpoints.items()},
            "state_components": sorted(self._statefuls),
            "files": files,
        }
        encoded = _json_bytes(manifest)
        _write_file(root / ".manifest.tmp", encoded)
        os.replace(root / ".manifest.tmp", root / _MANIFEST)
        _sync_directory(root)
        _sync_directory(root.parent)
        return SnapshotRef(
            str(root), manifest["snapshot_id"], hashlib.sha256(encoded).hexdigest()
        )

    @staticmethod
    def _invalidate(root: Path, error: BaseException) -> None:
        try:
            (root / _MANIFEST).unlink(missing_ok=True)
        except OSError as cleanup_error:
            error.add_note(f"Could not remove completion marker: {cleanup_error}")

    def _prepare_snapshot(self, directory: Path) -> _PreparedSnapshot:
        """Stage model files at the trainer's memory-safe point; do not publish."""
        if not self._lock.acquire(blocking=False):
            raise SnapshotError("A snapshot operation is already in progress")
        created = False
        try:
            if self._restore_failed:
                raise SnapshotError("Recreate participants after a failed restore")
            directory.mkdir(exist_ok=False)
            created = True
            self._save_checkpoints(directory)
            return _PreparedSnapshot(self, directory)
        except BaseException as exc:
            if created:
                self._invalidate(directory, exc)
            self._lock.release()
            if not isinstance(exc, Exception):
                raise
            raise SnapshotError(f"Snapshot preparation failed at {directory}") from exc

    def save_snapshot(
        self, directory: str | Path, *, context: SnapshotContext
    ) -> SnapshotRef:
        """Create a NEW snapshot; publish only after every participant finishes.

        The parent directory must already exist. On failure, the incomplete
        directory is retained without a manifest for diagnosis; retries must use
        a new directory. Large checkpoints are hashed in a streaming pass, whose
        I/O cost is part of snapshot creation. No periodic-save settings apply.
        """
        if not self._lock.acquire(blocking=False):
            raise SnapshotError("A snapshot operation is already in progress")
        root = None
        created = False
        try:
            if self._restore_failed:
                raise SnapshotError("Recreate participants after a failed restore")
            root = Path(directory).absolute()
            _context_dict(context)
            root.mkdir(exist_ok=False)
            created = True
            self._save_state(root)
            self._save_checkpoints(root)
            return self._publish(root, context)
        except BaseException as exc:
            if created and root is not None:
                self._invalidate(root, exc)
            if not isinstance(exc, Exception):
                raise
            raise SnapshotError(
                f"Snapshot save failed at {root}; do not use this directory"
            ) from exc
        finally:
            self._lock.release()

    def restore_snapshot(
        self,
        snapshot: SnapshotRef,
        *,
        expected_context: SnapshotContext,
        after_checkpoints: Callable[[SnapshotContext], None] | None = None,
    ) -> RestoreResult:
        """Strictly restore registered components from this exact snapshot.

        Verify every file, source binding and component BEFORE loading any role.
        Backend/layout admissibility is the Validator's responsibility. All
        components must already exist and remain quiescent through restoration.
        ``after_checkpoints`` synchronizes weights/versions before owner loads.
        Owner ``finalize_restore()`` hooks run last, in registration order, so
        initialization and synchronization cannot advance restored RNGs.
        The caller must perform runtime/weight-version confirmation before it
        resumes training; this method does not mark a target config as applied.
        """
        if not self._lock.acquire(blocking=False):
            raise SnapshotError("A snapshot operation is already in progress")
        loading = False
        try:
            if self._restore_failed:
                raise SnapshotError("Recreate participants after a failed restore")
            if after_checkpoints is not None and (
                not callable(after_checkpoints)
                or inspect.iscoroutinefunction(after_checkpoints)
            ):
                raise TypeError("after_checkpoints must be a blocking callback")
            manifest = inspect_snapshot(snapshot)
            if manifest["context"] != _context_dict(expected_context):
                raise SnapshotError(
                    "Snapshot does not match the expected source context"
                )
            formats = {k: v.format_id for k, v in self._checkpoints.items()}
            if manifest["checkpoints"] != formats:
                raise SnapshotError("Snapshot checkpoint roles or formats do not match")
            if manifest["state_components"] != sorted(self._statefuls):
                raise SnapshotError("Snapshot state components do not match")
            root = Path(snapshot.path)
            with (root / _STATE).open("rb") as stream:
                state = pickle.load(stream)
            if not isinstance(state, dict) or set(state) != set(self._statefuls):
                raise SnapshotError(
                    "Snapshot state payload does not match its manifest"
                )
            loading = True
            for role, participant in self._checkpoints.items():
                _call(participant.load, root / "checkpoints" / role)
            if after_checkpoints is not None:
                _call(after_checkpoints, SnapshotContext(**manifest["context"]))
            for name, owner in self._statefuls.items():
                _call(owner.load_state_dict, state[name])
            for owner in self._statefuls.values():
                if hasattr(owner, "finalize_restore"):
                    _call(owner.finalize_restore)
            return RestoreResult(
                snapshot,
                SnapshotContext(**manifest["context"]),
                tuple(self._checkpoints),
                tuple(self._statefuls),
            )
        except BaseException as exc:
            if loading:
                self._restore_failed = True
            if not isinstance(exc, Exception):
                raise
            detail = (
                "participants may be partially restored; recreate them"
                if loading
                else "preflight failed"
            )
            raise SnapshotError(f"Snapshot restore failed: {detail}") from exc
        finally:
            self._lock.release()


class _RestoreTrainingRuntime:
    """Apply the frozen mapping before the existing scheduler/RNG RPCs."""

    def __init__(
        self,
        owner: EngineRuntimeState,
        transition: ConfigurationTransition,
        snapshot: SnapshotRef,
    ):
        self.owner = owner
        self.transition = transition
        self.snapshot = snapshot

    def state_dict(self) -> str:
        return self.owner.state_dict()

    def load_state_dict(self, state: str) -> None:
        self.owner.load_state_dict(
            self.transition.map_training_state(state, self.snapshot)
        )

    def finalize_restore(self) -> None:
        self.owner.finalize_restore()


class _RestoreRollout:
    """Validate the native input binding while restored work is still fenced."""

    def __init__(
        self,
        rollout: Any,
        input_spec: dict[str, Any],
        transition: ConfigurationTransition | None = None,
    ):
        self.rollout = rollout
        self.input_spec = input_spec
        self.transition = transition

    def state_dict(self) -> dict[str, Any]:
        return self.rollout.state_dict()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        # Use the v1 controller's comparison (including tensor/array kwargs).
        # prepare_batch checks again, but that would be too late for queued work:
        # release_snapshot permits its dispatcher to submit pending inputs.
        from areal.infra.controller.rollout_controller import _same_input_spec

        if not _same_input_spec(state["input_spec"], self.input_spec):
            raise SnapshotError(
                "Rollout input bindings changed without a state mapping"
            )
        if self.transition is not None:
            state = self.transition.map_rollout_state(
                state, self.rollout.staleness_manager.state_dict()["limits"]
            )
        self.rollout.load_state_dict(state)


class TrainerSnapshotRestore:
    """One explicit restore into a fresh PPOTrainer, before its first batch.

    Construct before creating workers so corrupt or misbound snapshots fail
    without allocating a runtime. Call restore() after workflow/proxy setup and
    initial offload; call resume() only when there is a step left to execute.
    Any failure requires a new trainer. This is not a process restart manager.
    """

    def __init__(self, request: SnapshotRestoreRequest):
        if not isinstance(request, SnapshotRestoreRequest):
            raise TypeError("restore must be a SnapshotRestoreRequest")
        if not isinstance(request.expected_context, SnapshotContext):
            raise TypeError("expected_context must be a SnapshotContext")
        self.snapshot = copy.deepcopy(request.snapshot)
        self.context = SnapshotContext(**_context_dict(request.expected_context))
        self.transition = request.transition
        if self.transition is not None and (
            not isinstance(self.transition, ConfigurationTransition)
            or self.transition.source_configuration_id != self.context.configuration_id
        ):
            raise SnapshotError("Transition does not match the snapshot configuration")
        manifest = inspect_snapshot(self.snapshot)
        if manifest["context"] != _context_dict(self.context):
            raise SnapshotError("Snapshot does not match the expected source context")
        self.checkpoint_formats = manifest["checkpoints"]
        if (
            self.transition is not None
            and self.transition.training_layout is not None
            and self.transition.training_layout.backend == "megatron"
            and self.checkpoint_formats.get("actor") != MEGATRON_PORTABLE_FORMAT
        ):
            raise SnapshotError("Megatron layout changes require a portable snapshot")
        step = self.context.step_info
        if step["global_step"] != (
            step["epoch"] * step["steps_per_epoch"] + step["epoch_step"]
        ):
            raise SnapshotError("Snapshot training progress is inconsistent")
        self.next_step = step["global_step"] + 1
        self.attempted = False
        self.released = False
        self.result: RestoreResult | None = None

    def restore(
        self,
        trainer: Any,
        *,
        workflow: Any,
        workflow_kwargs: dict[str, Any] | None,
        should_accept_fn: Any,
        max_steps: int,
    ) -> RestoreResult:
        if self.attempted:
            raise SnapshotError(
                "Snapshot restore already attempted; create a new trainer"
            )
        self.attempted = True
        if len(trainer.train_dataloader) != self.context.step_info["steps_per_epoch"]:
            raise SnapshotError("Snapshot steps_per_epoch differs from the dataloader")
        checkpoints, statefuls = trainer._reconfiguration_components()
        # Old snapshots still use the ordinary DP-only checkpoint encoding.
        # Adapt this restore alone; the trainer's next save uses the new format.
        for role, participant in list(checkpoints.items()):
            if (
                self.checkpoint_formats.get(role) == "areal-megatron-dcp"
                and participant.format_id == MEGATRON_PORTABLE_FORMAT
            ):
                owner = statefuls.get(f"{role}_runtime")
                if not isinstance(owner, EngineRuntimeState):
                    raise SnapshotError(
                        "Legacy Megatron restore requires runtime state"
                    )

                def load_legacy(path: Path, engine: Any = owner.engine) -> None:
                    from areal.api.io_struct import SaveLoadMeta

                    _call(engine.load, SaveLoadMeta(str(path), "dcp", with_optim=True))

                checkpoints[role] = CheckpointParticipant(
                    "areal-megatron-dcp",
                    participant.save,
                    participant.wait_saved,
                    load_legacy,
                )
        if self.transition is not None and self.transition.training_layout is not None:
            owner = statefuls.get("actor_runtime")
            if not isinstance(owner, EngineRuntimeState):
                raise SnapshotError(
                    "Training layout restore requires actor runtime state"
                )
            statefuls["actor_runtime"] = _RestoreTrainingRuntime(
                owner, self.transition, self.snapshot
            )
        rollout = trainer.rollout
        gconfig = trainer.config.gconfig
        input_spec = dict(
            workflow=rollout._resolve_workflow_str(workflow),
            workflow_kwargs={} if workflow_kwargs is None else workflow_kwargs,
            should_accept_fn=rollout._resolve_should_accept_fn(should_accept_fn),
            group_size=gconfig.n_samples,
            reward_normalization=gconfig.reward_normalization,
            drop_incomplete_group=gconfig.drop_incomplete_group,
        )
        # A fresh dispatcher has no running work; this also establishes the
        # fence required by rollout.load_state_dict. Never release on failure.
        rollout.freeze_and_drain(timeout=60.0)
        rollout.pause()
        if trainer.eval_rollout is not None:
            trainer.eval_rollout.pause()
        awex = trainer._is_v1_awex_colocate(trainer.config)
        if awex:
            rollout.pause_generation_sync()
            rollout.offload(tags=["kv_cache"])
            rollout.offload(tags=["weights"])

        def synchronize(context: SnapshotContext) -> None:
            version = context.policy_version
            engines = [trainer.actor, trainer.critic, rollout, trainer.eval_rollout]
            # Disk rendezvous uses live versions, not WeightUpdateMeta.version.
            # A transfer under saved V would leave a marker that the first
            # resumed training update (also starting at V) could reuse. Reserve
            # a unique negative int64 while the fresh runtime is fenced; normal
            # policy versions are nonnegative. Keep the metadata's actual policy
            # version for the checkpoint path and inference adapter naming.
            transfer_version = version
            if trainer.weight_update_meta.type == "disk":
                transfer_version = -((uuid.uuid4().int & ((1 << 63) - 1)) + 1)
            trainer.actor.set_version(transfer_version)
            rollout.set_version(transfer_version)
            if awex:
                trainer.actor.onload()
            trainer.actor.update_weights(
                trainer.weight_update_meta.with_version(version)
            )
            for engine in engines:
                if engine is not None:
                    engine.set_version(version)
            # Eval shares inference servers with rollout: one weight transfer.
            if trainer._should_offload_actor:
                trainer._offload_model(trainer.actor, role="actor")
            if trainer._should_offload_critic:
                trainer._offload_model(trainer.critic, role="critic")
            if trainer._should_offload_rollout and self.next_step < max_steps:
                # Finish onload/generation setup BEFORE RNG finalizers. This
                # may resume the dispatcher, but its snapshot fence stays shut.
                trainer._onload_rollout()
            for engine in engines:
                if engine is not None and engine.get_version() != version:
                    raise SnapshotError("Restored policy version was not installed")

        manager = ReconfigurationManager(
            checkpoints=checkpoints,
            statefuls=dict(
                statefuls,
                rollout=_RestoreRollout(rollout, input_spec, self.transition),
            ),
        )
        self.result = manager.restore_snapshot(
            self.snapshot, expected_context=self.context, after_checkpoints=synchronize
        )
        return self.result

    def resume(self, trainer: Any) -> None:
        """Open restored admission only after all restore stages succeeded."""
        if self.result is None or self.released:
            raise SnapshotError("No newly restored snapshot to resume")
        if trainer.eval_rollout is not None:
            trainer.eval_rollout.resume()
        trainer.rollout.resume()
        trainer.rollout.release_snapshot()
        self.released = True


class _PreparedSnapshot:
    """Keep the manager locked between model save and end-of-step publication."""

    def __init__(self, manager: ReconfigurationManager, root: Path):
        self.manager = manager
        self.root = root
        self.closed = False

    def finish(self, context: SnapshotContext) -> SnapshotRef:
        if self.closed:
            raise SnapshotError("Prepared snapshot is already closed")
        try:
            _context_dict(context)
            self.manager._save_state(self.root)
            result = self.manager._publish(self.root, context)
        except BaseException as exc:
            self.abort(exc)
            if not isinstance(exc, Exception):
                raise
            raise SnapshotError(f"Snapshot publication failed at {self.root}") from exc
        self.closed = True
        self.manager._lock.release()
        return result

    def abort(self, error: BaseException) -> None:
        if not self.closed:
            self.manager._invalidate(self.root, error)
            self.closed = True
            self.manager._lock.release()


@dataclass(frozen=True)
class _SnapshotRequest:
    directory: Path
    state_id: str
    result: Future[SnapshotRef]
    stop_after_save: bool = False


class TrainerReconfiguration:
    """Optional, in-process snapshot request channel for one train() invocation.

    A trusted caller supplies source identities and requests snapshots; the
    trainer supplies the actual step and policy version. At most one request
    may be queued or executing. Requests arriving after the rollout preparation hook
    wait until the next step. A Future completes only after end-of-step state
    and the manifest are written. Training normally continues after a save.
    With ``stop_after_save=True``, the channel closes, rollout stays fenced,
    and train() returns a SnapshotStopResult at that completed boundary.

    Coverage remains registered_components. The trainer registers supplemental
    backend state, rollout queues, the input buffer and loader cursor. A snapshot
    Future confirms saving only: caller-owned trainer cleanup must finish before
    checking process exit. Restore/startup and restart orchestration are separate.

    Example (using the application's existing trainer and workflow)::

        control = TrainerReconfiguration(
            run_id=run_id, runtime_revision=revision, configuration_id=config_id
        )
        saved = control.request_snapshot(snapshot_dir, state_id="boundary-1")
        trainer.train(workflow=workflow, reconfiguration=control)
        snapshot = saved.result()

    For save-then-exit, request ``stop_after_save=True`` and call train() inside
    ``with PPOTrainer(...) as trainer``. Its normal context exit closes runtime
    resources; callers not using a context must call close() themselves.

    request_snapshot() may also be called by another thread during train().
    All other methods are called by the trainer thread; no planner or external
    process transport is installed by this class.
    """

    def __init__(
        self,
        *,
        run_id: str,
        runtime_revision: str,
        configuration_id: str,
        drain_timeout: float = 600.0,
    ):
        if not math.isfinite(drain_timeout) or drain_timeout <= 0:
            raise ValueError("Drain timeout must be finite and positive")
        self._drain_timeout = drain_timeout
        self._rollout: RolloutStateful | None = None
        self._identity = dict(
            run_id=run_id,
            runtime_revision=runtime_revision,
            configuration_id=configuration_id,
        )
        if any(
            not isinstance(v, str) or not v.strip() for v in self._identity.values()
        ):
            raise ValueError("Snapshot source identities must be nonempty strings")
        self._lock = threading.Lock()
        self._manager: ReconfigurationManager | None = None
        self._pending: _SnapshotRequest | None = None
        self._active: _SnapshotRequest | None = None
        self._prepared: _PreparedSnapshot | None = None
        self._step_info: dict[str, int] | None = None
        self._closed = False

    def request_snapshot(
        self,
        directory: str | Path,
        *,
        state_id: str,
        stop_after_save: bool = False,
    ) -> Future[SnapshotRef]:
        """Queue a snapshot, optionally ending training after successful saving.

        The directory must be new and its parent must exist. The stop flag is
        fixed for this request; cancelling before the trainer latches it also
        cancels the stop. The returned Future confirms saving, not process exit.
        """
        if not isinstance(state_id, str) or not state_id.strip():
            raise ValueError("Snapshot state_id must be a nonempty string")
        if type(stop_after_save) is not bool:
            raise TypeError("stop_after_save must be a bool")
        request = _SnapshotRequest(
            Path(directory).absolute(), state_id, Future(), stop_after_save
        )
        with self._lock:
            if self._closed:
                raise SnapshotError("This trainer reconfiguration channel is closed")
            if self._pending is not None and self._pending.result.cancelled():
                # Notify wait()/as_completed() before discarding the cancelled job.
                self._pending.result.set_running_or_notify_cancel()
                self._pending = None
            if self._pending is not None or self._active is not None:
                raise SnapshotError("A snapshot request is already pending or running")
            self._pending = request
        return request.result

    def bind(
        self,
        *,
        checkpoints: Mapping[str, CheckpointParticipant],
        statefuls: Mapping[str, Stateful],
        rollout: RolloutStateful | None = None,
    ) -> None:
        """Register this trainer's actual state owners once, before training."""
        with self._lock:
            if self._closed or self._manager is not None:
                raise SnapshotError(
                    "Use a new reconfiguration channel for each train()"
                )
            if rollout is not None:
                if "rollout" in statefuls:
                    raise ValueError("Rollout state has two owners")
                statefuls = dict(statefuls, rollout=rollout)
            self._rollout = rollout
            self._manager = ReconfigurationManager(
                checkpoints=checkpoints, statefuls=statefuls
            )

    @property
    def has_active_snapshot(self) -> bool:
        with self._lock:
            return self._active is not None

    def prepare_step_snapshot(self, *, step_info: dict[str, int]) -> None:
        """Latch a request while inference can still run, then drain rollout."""
        with self._lock:
            if self._closed or self._manager is None:
                raise SnapshotError(
                    "Snapshot channel is not bound to an active trainer"
                )
            if self._active is not None:
                raise SnapshotError("The previous step snapshot is still active")
            request = self._pending
            self._pending = None
            if request is None or not request.result.set_running_or_notify_cancel():
                return
            self._active = request
            self._step_info = dict(step_info)
        try:
            if self._rollout is not None:
                self._rollout.freeze_and_drain(timeout=self._drain_timeout)
        except BaseException as exc:
            self.close(exc)
            raise

    def begin_step_snapshot(self, *, step_info: dict[str, int]) -> None:
        """Save model state at the native checkpoint point after early draining."""
        # Preserve the standalone channel API when no rollout owner is bound.
        if self._rollout is None:
            self.prepare_step_snapshot(step_info=step_info)
        if self._active is None:
            return
        try:
            if self._prepared is not None or self._step_info != step_info:
                raise SnapshotError(
                    "Snapshot model hook does not match its prepared step"
                )
            self._prepared = self._manager._prepare_snapshot(self._active.directory)
        except BaseException as exc:
            self.close(exc)
            raise

    def finish_step_snapshot(self, *, policy_version: int) -> SnapshotStopResult | None:
        """Publish the latched boundary; return an exit result only for stop requests."""
        if self._active is None:
            return
        request = self._active
        try:
            if self._prepared is None or self._step_info is None:
                raise SnapshotError("No prepared model checkpoint for this step")
            if (
                self._rollout is not None
                and self._rollout.get_version() != policy_version
            ):
                raise SnapshotError("Actor and rollout policy versions disagree")
            context = SnapshotContext(
                **self._identity,
                state_id=request.state_id,
                step_info=self._step_info,
                policy_version=policy_version,
            )
            result = self._prepared.finish(context)
        except BaseException as exc:
            self.close(exc)
            raise
        stop = SnapshotStopResult(result, context) if request.stop_after_save else None
        if self._rollout is not None and stop is None:
            self._rollout.release_snapshot()
        with self._lock:
            self._active = None
            self._prepared = None
            self._step_info = None
            if stop is not None:
                # Terminal before invoking callbacks: no subsequent request can
                # race with trainer exit or reopen rollout admission.
                self._closed = True
        # Ordinary save callbacks may request another step; stop callbacks cannot.
        request.result.set_result(result)
        return stop

    def close(self, error: BaseException | None = None) -> None:
        """End requests on training exit, including failures between the two hooks."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            requests = (self._active, self._pending)
            self._active = self._pending = None
        failure = (
            error
            if error is not None
            else SnapshotError(
                "Training ended before the requested snapshot could be completed"
            )
        )
        if self._prepared is not None:
            self._prepared.abort(failure)
            self._prepared = None
        active, pending = requests
        if active is not None and not active.result.done():
            active.result.set_exception(failure)
        if pending is not None and pending.result.set_running_or_notify_cancel():
            pending.result.set_exception(failure)
