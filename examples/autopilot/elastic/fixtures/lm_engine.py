# SPDX-License-Identifier: Apache-2.0

"""Real FSDP engine with diagnostic RPCs for the lifecycle experiment only."""

import json
import os
from pathlib import Path

import psutil
import torch

from examples.autopilot.elastic.fixtures.diagnostics import record, rng_probe

from areal.autopilot.elastic.runtime.state.checkpoint import (
    atomic_json,
    capture_runtime_state,
    recovery_digest,
    restore_recovery_runtime,
    restore_runtime_state,
)
from areal.engine.fsdp_engine import FSDPEngine, FSDPPPOActor
from areal.utils import seeding


class LifecycleEngine(FSDPEngine):
    # Reuse AReaL's existing LM-loss path: real backward/AdamW, no PPO claim.
    fixed_batch_update = FSDPPPOActor.sft_train_batch
    _record = record
    rng_probe = rng_probe

    def initialize(self, addr, ft_spec, *, seed=17):
        seeding.set_random_seed(seed, key=f"elastic-rank-{self.rank}")
        torch.use_deterministic_algorithms(True)
        super().initialize(addr, ft_spec)
        if self.config._version == "v2":
            self._record(
                "process",
                {
                    "pid": os.getpid(),
                    "created": psutil.Process().create_time(),
                    "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                },
            )

    def train_lm(self, data):
        # Called through the real v2 gateway -> data proxy -> worker SFT route.
        probe = self.rng_probe()
        stats = self.fixed_batch_update(data)
        self._record(
            "update", {"rng_probe": probe, "stats": stats}, step=self.get_version() + 1
        )
        return stats

    def save(self, meta):
        if self.config._version != "v2":
            return super().save(meta)
        state = self.capture_runtime()
        super().save(meta)
        atomic_json(Path(meta.path).parent / f"runtime-{self.rank}.json", state)

    def load(self, meta):
        super().load(meta)
        if self.config._version == "v2":
            state = json.loads(
                (Path(meta.path).parent / f"runtime-{self.rank}.json").read_text()
            )
            restored = restore_recovery_runtime(self, state)
            self._record(
                "restored",
                {"state": restored, "recovery_sha256": recovery_digest(state)},
            )

    def capture_runtime(self):
        return capture_runtime_state(self)

    def set_training_mode(self):
        # FSDPEngine.train() returns self, which is not an RPC result value.
        self.train()

    def restore_runtime(self, state):
        restore_runtime_state(self, state)
        return self.capture_runtime()
