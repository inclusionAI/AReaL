# SPDX-License-Identifier: Apache-2.0

"""Managed Rollout generations behind the same whole-episode admission gate."""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path

from areal.autopilot.elastic.runtime.state.checkpoint import artifact_manifest


class RolloutGroup:
    """Own a service factory; restart only after all gated sessions are removed.

    The local SGLang profile maps sequence capacity to max_running_requests,
    batch tokens to max_prefill_tokens, and KV tokens to max_total_tokens.
    max_inflight_batches stays one under the current strict training loop.
    """

    def __init__(
        self,
        *,
        config,
        model: Path,
        model_name: str,
        model_revision: str,
        root: Path,
        lock_root: Path,
        service_factory,
        context_length: int = 512,
    ):
        self.config, self.model, self.model_name = config, model.resolve(), model_name
        self.model_revision = model_revision
        if not isinstance(model_revision, str) or not model_revision:
            raise ValueError("Pinned Rollout model revision is required")
        self.base_files = artifact_manifest(self.model)
        self.root, self.lock_root = root.resolve(), lock_root.resolve()
        self.factory, self.context_length = service_factory, context_length
        self.service = None
        self.devices = ()
        self.weights_version = None
        self._identity = {
            name: getattr(config, name)
            for name in (
                "model_id",
                "hardware_id",
                "runtime_id",
                "sampling_id",
                "backend",
            )
        }
        context = {
            "model": str(self.model),
            "model_revision": model_revision,
            "base_files": self.base_files,
            "identity": self._identity,
            "root": str(self.root),
            "context_length": context_length,
            "factory": f"{service_factory.__module__}.{service_factory.__qualname__}",
        }
        self.context_id = hashlib.sha256(
            json.dumps(context, sort_keys=True).encode()
        ).hexdigest()
        self.validate(config)

    def validate(self, target):
        if any(
            getattr(target, name) != value for name, value in self._identity.items()
        ):
            raise ValueError(
                "Rollout model/hardware/runtime/sampling identity is fixed"
            )
        if target.backend != "sglang" or target.max_inflight_batches != 1:
            raise ValueError(
                "Local SGLang requires one inflight batch under the episode gate"
            )
        if (
            min(target.max_batch_tokens, target.kv_capacity_tokens)
            < self.context_length
        ):
            raise ValueError("Rollout token limits must cover the fixed context length")
        model = json.loads((self.model / "config.json").read_text())
        for field in ("num_attention_heads", "num_key_value_heads"):
            count = model.get(field, model.get("num_attention_heads"))
            if not isinstance(count, int) or count % target.tp:
                raise ValueError(f"Rollout TP does not divide {field}")

    @property
    def controller(self):
        if self.service is None:
            raise RuntimeError("Rollout service is not running")
        return self.service.controller

    @property
    def admin_key(self):
        return self.service.admin_key

    @property
    def owner(self):
        return self.service.owner

    def start(self, config, devices):
        self.validate(config)
        if artifact_manifest(self.model) != self.base_files:
            raise ValueError("Immutable Rollout base model changed")
        if self.service is not None:
            raise RuntimeError(
                "Verify old Rollout cleanup before starting a generation"
            )
        if len(devices) != config.world_size or len(set(devices)) != len(devices):
            raise ValueError("Rollout placement differs from replica/TP layout")
        self.config, self.devices, self.weights_version = config, tuple(devices), None
        self.service = self.factory(
            model=self.model,
            model_name=self.model_name,
            devices=list(devices),
            rollout_config=config,
            context_length=self.context_length,
            root=self.root / uuid.uuid4().hex,
            lock_root=self.lock_root,
        )
        self.service.start()

    def assert_drained(self):
        if self.service is not None:
            self.service.assert_drained()

    def quiesce(self):
        self.assert_drained()
        self.controller.pause()
        self.controller.pause_generation()

    def mark_weights_verified(self, version):
        # A new controller's capacity counters must start at the recovered version.
        self.controller.staleness_manager.on_version_recovered(version)
        self.weights_version = version

    def verified(self, version):
        return (
            self.service is not None
            and self.weights_version == version
            and self.service.verify_version(version)
        )

    def resume(self):
        self.service.resume()

    def stop(self):
        if self.service is not None:
            self.service.close()  # On failure retain the owner and all GPU locks.
            self.service = None
        self.weights_version = None
