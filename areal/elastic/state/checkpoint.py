# SPDX-License-Identifier: Apache-2.0

"""Commit recoverable training state and validate checkpoint weight artifacts.

Model/optimizer serialization uses AReaL's engine save/load APIs. This module adds
atomic generations, queue-cursor binding, scheduler/RNG recovery and export checks.
"""

import hashlib
import json
import os
import random
import shutil
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from areal.utils.atomic import atomic_json


class CheckpointStore:
    def __init__(self, root: Path, *, keep: int = 2, free_space_floor: int = 10 << 30):
        if keep < 1:
            raise ValueError("keep must be at least one")
        self.root = root.resolve()
        self.keep = keep
        self.free_space_floor = free_space_floor
        self.root.mkdir(parents=True, exist_ok=True)

    def commit(
        self,
        *,
        step: int,
        world_size: int,
        compatibility: dict[str, Any],
        write: Callable[[Path], None],
        estimated_bytes: int,
        data_cursor: dict[str, Any] | None = None,
        data_progress: dict[str, int] | None = None,
    ) -> Path:
        if step < 1 or world_size < 1 or estimated_bytes < 0:
            raise ValueError("Invalid checkpoint step, world size or size estimate")
        if shutil.disk_usage(self.root).free < self.free_space_floor + estimated_bytes:
            raise OSError("Insufficient space to commit another checkpoint safely")
        if data_cursor is not None:
            self._validate_cursor(data_cursor, step)
        previous = None
        if (self.root / "latest.json").exists():
            previous_path, previous_manifest = self.latest(compatibility)
            if step < previous_manifest["global_step"]:
                raise ValueError("Checkpoint cannot move published progress backwards")
            previous = previous_path.name
        # A crash after directory rename but before latest publication leaves an
        # orphan. A unique generation lets replay commit the same step again.
        target = self.root / f"step-{step:08d}-{uuid.uuid4().hex}"
        staging = self.root / f".step-{step:08d}-{uuid.uuid4().hex}.pending"
        staging.mkdir()
        # Failed/incomplete writes remain pending and are never selected by latest.
        write(staging)
        if not (staging / "dcp" / ".metadata").is_file():
            raise ValueError("Missing DCP metadata")
        for rank in range(world_size):
            state = json.loads((staging / f"runtime-{rank}.json").read_text())
            if (state["rank"], state["world_size"], state["policy_version"]) != (
                rank,
                world_size,
                step,
            ):
                raise ValueError("Incomplete or inconsistent rank runtime state")
        for file in staging.rglob("*"):
            if file.is_file():
                with file.open("rb") as stream:
                    os.fsync(stream.fileno())
        # Persist nested DCP directory entries before publishing the generation.
        for directory in sorted(
            (path for path in staging.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        manifest = {
            "format_version": 1,
            "global_step": step,
            "next_batch_id": step,
            "world_size": world_size,
            "compatibility": compatibility,
            "previous_generation": previous,
        }
        if data_cursor is not None:
            manifest.pop("next_batch_id")
            manifest["data_cursor"] = data_cursor
        if data_progress is not None:
            if any(
                type(data_progress.get(key)) is not int or data_progress[key] < 0
                for key in ("samples", "effective_tokens")
            ):
                raise ValueError("Invalid logical data progress")
            manifest["data_progress"] = data_progress
        atomic_json(staging / "manifest.json", manifest)
        os.replace(staging, target)
        atomic_json(self.root / "latest.json", {"generation": target.name})
        # Follow published history, not directory timestamps: a newer orphan is
        # not a substitute for the previous recoverable generation.
        retained = {target.name}
        while previous and len(retained) < self.keep:
            if Path(previous).name != previous or not previous.startswith("step-"):
                raise ValueError("Invalid checkpoint history")
            prior = self.root / previous / "manifest.json"
            if not prior.is_file() or previous in retained:
                break
            retained.add(previous)
            previous = json.loads(prior.read_text()).get("previous_generation")
        for old in self.root.glob("step-*"):
            if old.name not in retained and (old / "manifest.json").is_file():
                shutil.rmtree(old)
        return target

    @staticmethod
    def _validate_cursor(cursor: dict[str, Any], step: int) -> None:
        if (
            cursor.get("format_version") != 1
            or not isinstance(cursor.get("queue_id"), str)
            or not cursor["queue_id"]
            or type(cursor.get("applied_batches")) is not int
            or cursor["applied_batches"] != step
            or type(cursor.get("through_sequence")) is not int
            or cursor["through_sequence"] < step
            or not isinstance(cursor.get("chain_sha256"), str)
            or len(cursor["chain_sha256"]) != 64
        ):
            raise ValueError("Checkpoint data cursor disagrees with training progress")

    def latest(self, compatibility: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
        pointer = json.loads((self.root / "latest.json").read_text())
        name = pointer["generation"]
        if Path(name).name != name or not name.startswith("step-"):
            raise ValueError("Invalid checkpoint generation")
        path = self.root / name
        manifest = json.loads((path / "manifest.json").read_text())
        if (
            manifest["format_version"] != 1
            or manifest["compatibility"] != compatibility
        ):
            raise ValueError("Checkpoint is incompatible with this training run")
        if not (path / "dcp" / ".metadata").is_file():
            raise ValueError("Checkpoint DCP payload is missing")
        if "data_cursor" in manifest:
            self._validate_cursor(manifest["data_cursor"], manifest["global_step"])
        elif manifest["next_batch_id"] != manifest["global_step"]:
            raise ValueError(
                "Checkpoint data position disagrees with training progress"
            )
        for rank in range(manifest["world_size"]):
            if not (path / f"runtime-{rank}.json").is_file():
                raise ValueError("Checkpoint rank runtime state is missing")
            state = json.loads((path / f"runtime-{rank}.json").read_text())
            if (state["rank"], state["world_size"], state["policy_version"]) != (
                rank,
                manifest["world_size"],
                manifest["global_step"],
            ):
                raise ValueError("Checkpoint rank state disagrees with manifest")
        return path, manifest


def capture_runtime_state(engine: Any) -> dict[str, Any]:
    import numpy as np
    import torch

    np_state = np.random.get_state()
    return {
        "format_version": 1,
        "rank": engine.rank,
        "world_size": engine.world_size,
        "policy_version": engine.get_version(),
        "lr_scheduler": engine.lr_scheduler.state_dict(),
        "python_rng": random.getstate(),
        "numpy_rng": [np_state[0], np_state[1].tolist(), *np_state[2:]],
        "torch_rng": torch.get_rng_state().tolist(),
        "cuda_rng": (
            torch.cuda.get_rng_state(engine.device).cpu().tolist()
            if engine.device.type == "cuda"
            else None
        ),
    }


def restore_runtime_state(engine: Any, state: dict[str, Any]) -> None:
    if state["format_version"] != 1:
        raise ValueError("Unsupported runtime state format")
    if (state["rank"], state["world_size"]) != (engine.rank, engine.world_size):
        raise ValueError("Runtime state rank/world size mismatch")
    needs_cuda = engine.device.type == "cuda"
    if needs_cuda != (state["cuda_rng"] is not None):
        raise ValueError("Runtime state device type mismatch")
    engine.lr_scheduler.load_state_dict(state["lr_scheduler"])
    expected_lrs = engine.lr_scheduler.get_last_lr()
    actual_lrs = [g["lr"] for g in engine.optimizer.param_groups]
    if actual_lrs != expected_lrs:
        raise ValueError(
            f"Optimizer/scheduler learning rates disagree: {actual_lrs} != {expected_lrs}"
        )
    engine.set_version(state["policy_version"])
    restore_random_state(engine, state)


def restore_random_state(engine: Any, state: dict[str, Any]) -> None:
    """Restore per-rank RNG without assuming a backend's scheduler interface."""
    import numpy as np
    import torch

    if (state["rank"], state["world_size"]) != (engine.rank, engine.world_size):
        raise ValueError("Runtime state rank/world size mismatch")
    needs_cuda = engine.device.type == "cuda"
    if needs_cuda != (state["cuda_rng"] is not None):
        raise ValueError("Runtime state device type mismatch")
    py_state = state["python_rng"]
    random.setstate((py_state[0], tuple(py_state[1]), py_state[2]))
    np_state = state["numpy_rng"]
    np.random.set_state(
        (np_state[0], np.asarray(np_state[1], dtype=np.uint32), *np_state[2:])
    )
    torch.set_rng_state(torch.tensor(state["torch_rng"], dtype=torch.uint8))
    if needs_cuda:
        torch.cuda.set_rng_state(
            torch.tensor(state["cuda_rng"], dtype=torch.uint8), engine.device
        )


WEIGHT_TYING = "preserve-config-v1"


def artifact_manifest(root: Path) -> dict:
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            with path.open("rb") as stream:
                files[str(path.relative_to(root))] = hashlib.file_digest(
                    stream, "sha256"
                ).hexdigest()
    return files


def tie_model_embeddings(model) -> None:
    """Honor tied HF configs before FSDP, even if a loader materialized both keys."""
    if model.config.tie_word_embeddings:
        model.get_output_embeddings().weight = model.get_input_embeddings().weight
        if hasattr(model, "get_expanded_tied_weights_keys"):
            model.all_tied_weights_keys = model.get_expanded_tied_weights_keys(
                all_submodels=True
            )


def verify_tied_export(root: Path) -> None:
    """FSDP's per-key BF16 cast may duplicate aliases; their values must agree."""
    import torch
    from safetensors import safe_open

    if not json.loads((root / "config.json").read_text()).get("tie_word_embeddings"):
        return
    index_path = root / "model.safetensors.index.json"
    if index_path.exists():
        files = json.loads(index_path.read_text())["weight_map"]
    else:
        with safe_open(
            root / "model.safetensors", framework="pt", device="cpu"
        ) as weights:
            files = dict.fromkeys(weights.keys(), "model.safetensors")
    if "lm_head.weight" not in files:
        return
    values = []
    for name in ("model.embed_tokens.weight", "lm_head.weight"):
        with safe_open(root / files[name], framework="pt", device="cpu") as weights:
            values.append(weights.get_tensor(name))
    torch.testing.assert_close(*values, rtol=0, atol=0)


def recovery_digest(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def recovery_view(
    source: Path, manifest: dict[str, Any], *, target_world_size: int, root: Path
) -> Path:
    """Create a loading view without modifying a committed source generation."""
    source_world = manifest["world_size"]
    if type(source_world) is not int or source_world < 1:
        raise ValueError("Invalid recovery source world size")
    states = [
        json.loads((source / f"runtime-{rank}.json").read_text())
        for rank in range(source_world)
    ]
    for rank, state in enumerate(states):
        if state.get("format_version") != 1:
            raise ValueError("Invalid recovery source runtime format")
        if (state["rank"], state["world_size"], state["policy_version"]) != (
            rank,
            source_world,
            manifest["global_step"],
        ):
            raise ValueError("Recovery source rank identity mismatch")
        if state["lr_scheduler"] != states[0]["lr_scheduler"]:
            raise ValueError("Recovery source scheduler differs across ranks")
    if type(target_world_size) is not int or target_world_size < 1:
        raise ValueError("Invalid recovery target world size")
    if (
        source_world != target_world_size
        and manifest["compatibility"].get("elastic_dp") != "fsdp-pure-dp-v1"
    ):
        raise ValueError("Checkpoint does not authorize data-parallel resizing")
    view = root / f"load-{uuid.uuid4().hex}"
    view.mkdir(parents=True)
    (view / "dcp").symlink_to((source / "dcp").resolve(), target_is_directory=True)
    source_key = recovery_digest({"generation": source.name, "manifest": manifest})
    for rank in range(target_world_size):
        if source_world == target_world_size:
            state = states[rank]
        else:
            state = {
                "format_version": 2,
                "rng_rule": "elastic-dp-reseed-v1",
                "source_sha256": source_key,
                "run_seed": manifest["compatibility"]["seed"],
                "source_world_size": source_world,
                "rank": rank,
                "world_size": target_world_size,
                "policy_version": manifest["global_step"],
                "lr_scheduler": states[0]["lr_scheduler"],
            }
        atomic_json(view / f"runtime-{rank}.json", state)
    atomic_json(
        view / "recovery.json",
        {
            "source": source.name,
            "source_sha256": source_key,
            "target_world_size": target_world_size,
        },
    )
    return view


def restore_recovery_runtime(engine: Any, state: dict[str, Any]) -> dict[str, Any]:
    """Exact same-topology RNG restore, deterministic fresh RNG after DP resize.

    Native DCP restores model and Adam state before this helper is called. RNG
    reseeding does not promise bitwise equivalence across different DP layouts.
    """
    if state["format_version"] == 1:
        target = state
    elif (
        state["format_version"] == 2 and state.get("rng_rule") == "elastic-dp-reseed-v1"
    ):
        import numpy as np
        import torch

        digest = state.get("source_sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or type(state.get("run_seed")) is not int
            or type(state.get("source_world_size")) is not int
            or state["source_world_size"] < 1
            or type(state.get("world_size")) is not int
            or state["world_size"] < 1
            or state["source_world_size"] == state["world_size"]
            or type(state.get("rank")) is not int
            or not 0 <= state["rank"] < state["world_size"]
            or type(state.get("policy_version")) is not int
            or state["policy_version"] < 1
            or not isinstance(state.get("lr_scheduler"), dict)
        ):
            raise ValueError("Incomplete or invalid recovery plan")
        if (state["rank"], state["world_size"]) != (engine.rank, engine.world_size):
            raise ValueError("Recovery plan rank/world size mismatch")
        seed = int(recovery_digest(state)[:8], 16)
        np_state = np.random.RandomState(seed).get_state()
        target = {
            "format_version": 1,
            "rank": engine.rank,
            "world_size": engine.world_size,
            "policy_version": state["policy_version"],
            "lr_scheduler": state["lr_scheduler"],
            "python_rng": random.Random(seed).getstate(),
            "numpy_rng": [np_state[0], np_state[1].tolist(), *np_state[2:]],
            "torch_rng": torch.Generator(device="cpu")
            .manual_seed(seed)
            .get_state()
            .tolist(),
            "cuda_rng": (
                torch.Generator(device=engine.device)
                .manual_seed(seed)
                .get_state()
                .cpu()
                .tolist()
                if engine.device.type == "cuda"
                else None
            ),
        }
    else:
        raise ValueError("Unsupported recovery runtime format")
    restore_runtime_state(engine, target)
    restored = capture_runtime_state(engine)
    if json.loads(json.dumps(restored)) != json.loads(json.dumps(target)):
        raise RuntimeError("Recovered runtime differs from the recovery plan")
    return restored
