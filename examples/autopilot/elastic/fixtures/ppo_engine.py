# SPDX-License-Identifier: Apache-2.0

"""Real PPO worker with audit records and recoverable DCP runtime state."""

import dataclasses
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
    tie_model_embeddings,
)
from areal.engine.fsdp_engine import FSDPPPOActor
from areal.utils import seeding


class ElasticPPOEngine(FSDPPPOActor):
    _record = record
    rng_probe = rng_probe

    def capture_runtime(self):
        return capture_runtime_state(self)

    def restore_runtime(self, state):
        restore_runtime_state(self, state)
        return self.capture_runtime()

    def _create_device_model(self):
        super()._create_device_model()
        # Transformers 5 can materialize both checkpoint aliases independently.
        # Fix ownership before fully_shard builds parameters and optimizer state.
        tie_model_embeddings(self.model)

    def initialize(self, addr, ft_spec, *, seed=17):
        seeding.set_random_seed(seed, key=f"elastic-rank-{self.rank}")
        torch.use_deterministic_algorithms(True)
        super().initialize(addr, ft_spec)
        shared_embeddings = (
            self.model.get_input_embeddings().weight
            is self.model.get_output_embeddings().weight
        )
        if self.model.config.tie_word_embeddings and not shared_embeddings:
            raise RuntimeError("FSDP lost the configured embedding/output-head alias")
        self._record(
            "process",
            {
                "pid": os.getpid(),
                "created": psutil.Process().create_time(),
                "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                "shared_embeddings": shared_embeddings,
            },
        )

    def _parameter_probe(self):
        parameter = dict(self.model.named_parameters())[
            "model.layers.0.mlp.down_proj.weight"
        ]
        local = parameter.to_local() if hasattr(parameter, "to_local") else parameter
        return local.detach().reshape(-1)[:4096].float().cpu().clone()

    def optimizer_step(self):
        result = super().optimizer_step()
        self._optimizer_results.append(result)
        return result

    def ppo_update(self, data):
        if len(data) != 1 or data[0]["policy_version"] != self.get_version():
            raise ValueError("Expected one current-policy JSON batch")
        raw = data[0]["trajectory"]
        dtypes = {
            "input_ids": torch.long,
            "versions": torch.long,
            "turn_ids": torch.long,
            "attention_mask": torch.bool,
        }
        tensors = {
            name: torch.tensor(
                value, dtype=dtypes.get(name, torch.float32), device=self.device
            )
            for name, value in raw.items()
        }
        before = self._parameter_probe()
        probe = self.rng_probe()
        prepared = self.compute_advantages([tensors])
        advantages = prepared[0]["advantages"][prepared[0]["loss_mask"].bool()]
        advantage_range = [float(advantages.min()), float(advantages.max())]
        self._optimizer_results = []
        super().ppo_update(prepared)
        after = self._parameter_probe()
        self._record(
            "ppo",
            {
                "rng_probe": probe,
                "optimizer": self._optimizer_results[-1],
                "optimizer_updates": len(self._optimizer_results),
                "parameter_max_delta": float((after - before).abs().max()),
                "advantage_range": advantage_range,
                "original_rewards": raw["rewards"],
                "behavior_version": data[0]["policy_version"],
            },
            step=self.get_version() + 1,
        )

    def save(self, meta):
        if meta.weight_format == "hf":
            return super().save(dataclasses.replace(meta, tokenizer=self.tokenizer))
        state = self.capture_runtime()
        super().save(meta)
        atomic_json(Path(meta.path).parent / f"runtime-{self.rank}.json", state)

    def load(self, meta):
        super().load(meta)
        if meta.weight_format == "dcp":
            shared_embeddings = (
                self.model.get_input_embeddings().weight
                is self.model.get_output_embeddings().weight
            )
            if self.model.config.tie_word_embeddings and not shared_embeddings:
                raise RuntimeError(
                    "Checkpoint restore lost the embedding/output-head alias"
                )
            state = json.loads(
                (Path(meta.path).parent / f"runtime-{self.rank}.json").read_text()
            )
            restored = restore_recovery_runtime(self, state)
            self._record(
                "restored",
                {
                    "state": restored,
                    "shared_embeddings": shared_embeddings,
                    "recovery_sha256": recovery_digest(state),
                },
            )
