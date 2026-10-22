# SPDX-License-Identifier: Apache-2.0

"""Per-rank scheduler, version and RNG state for native training checkpoints."""

import random
from typing import Any


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
