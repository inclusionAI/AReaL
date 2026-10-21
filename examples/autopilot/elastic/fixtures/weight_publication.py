# SPDX-License-Identifier: Apache-2.0

"""SGLang fixture publishing checkpoint weights before admitting agent episodes."""

import hashlib
import json
import threading
from contextlib import contextmanager
from pathlib import Path

import httpx

from areal.autopilot.elastic.runtime.state.checkpoint import (
    WEIGHT_TYING,
    artifact_manifest,
    atomic_json,
)

PROBE_PARAMETER = "model.layers.0.mlp.down_proj.weight"
PROBE_PARAMETERS = (PROBE_PARAMETER, "model.embed_tokens.weight")


class WeightPublisher:
    """Single producer gate; serialize whole episodes with publication.

    External clients bypassing episode() are outside this prototype's contract.
    A failed publish leaves admission closed. Retry the committed artifact, never
    replay an already checkpointed optimizer update to repair publication.
    """

    def __init__(self, service, *, state: Path):
        self.service = service
        self.state = state
        self.state.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.ready = False

    @contextmanager
    def episode(self):
        with self.lock:
            if not self.ready:
                raise RuntimeError(
                    "Rollout admission is closed pending weight publication"
                )
            yield self.service.controller.get_version()

    def initialize(self, checkpoint: Path | None = None, *, admit: bool = True):
        # Each new inference incarnation reloads latest, regardless of receipts
        # left by a previous process. Version labels alone are not evidence.
        if checkpoint is not None:
            return self.publish(checkpoint, admit=admit)
        if self.service.controller.get_version() != 0:
            raise RuntimeError("Expected the immutable initial model at version zero")
        model = getattr(self.service, "model", None)
        if isinstance(model, Path):
            with httpx.Client(timeout=180, trust_env=False) as client:
                for index, backend in enumerate(self.service.controller._inf_addrs):
                    self._verify_parameters(client, backend, model, index)
        if not admit:
            self.service.controller.pause()
            self.service.controller.pause_generation()
        self._mark_verified(0)
        if admit:
            self._resume()
        self.ready = admit
        return {"version": 0, "initial_model": True}

    def publish(self, checkpoint: Path, *, admit: bool = True) -> dict:
        with self.lock:
            self.ready = False
            manifest = json.loads((checkpoint / "manifest.json").read_text())
            serving = json.loads((checkpoint / "serving.json").read_text())
            if (
                manifest.get("compatibility", {}).get("weight_tying") != WEIGHT_TYING
                or serving.get("weight_tying") != WEIGHT_TYING
            ):
                raise ValueError("Checkpoint has no verified embedding weight tying")
            version = manifest["global_step"]
            if serving["version"] != version or serving["files"] != artifact_manifest(
                checkpoint / "hf"
            ):
                raise ValueError("Serving artifact disagrees with committed checkpoint")
            controller = self.service.controller
            controller.pause()
            controller.pause_generation()
            record = {
                "generation": checkpoint.name,
                "version": version,
                "artifact_digest": hashlib.sha256(
                    json.dumps(serving, sort_keys=True).encode()
                ).hexdigest(),
                "inference_incarnation": self.service.owner,
                "verified_parameters": list(PROBE_PARAMETERS),
                "verification_scope": "rank-zero shard probes and all-backend reload/version checks",
            }
            # No automatic continue on failure. Callers may retry this exact target.
            with httpx.Client(timeout=180, trust_env=False) as client:
                for index, backend in enumerate(controller._inf_addrs):
                    response = client.post(
                        f"{backend}/update_weights_from_disk",
                        json={
                            "model_path": str(checkpoint / "hf"),
                            "abort_all_requests": True,
                        },
                    )
                    response.raise_for_status()
                    if response.json().get("success") is not True:
                        raise RuntimeError(
                            "SGLang rejected weight reload despite HTTP success"
                        )
                    self._verify_parameters(client, backend, checkpoint, index)
                controller.set_version(version)
                headers = {"Authorization": f"Bearer {self.service.admin_key}"}
                for proxy in controller._data_proxy_addrs:
                    response = client.get(f"{proxy}/get_version", headers=headers)
                    response.raise_for_status()
                    if response.json().get("version") != version:
                        raise RuntimeError(
                            "Data proxy version does not match loaded weights"
                        )
            atomic_json(self.state, record)
            self._mark_verified(version)
            if admit:
                self._resume()
                self.ready = True
            return {**record, "parameters_verified": True}

    def _mark_verified(self, version):
        callback = getattr(self.service, "mark_weights_verified", None)
        if callback is not None:
            callback(version)

    def _resume(self):
        callback = getattr(self.service, "resume", None)
        if callback is not None:
            callback()
        else:
            self.service.controller.continue_generation()
            self.service.controller.resume()

    def _verify_parameters(self, client, backend: str, checkpoint: Path, index: int):
        import torch
        from safetensors import safe_open

        saved = self.state.parent / f"loaded-{self.service.owner}-{index}.pt"
        try:
            response = client.post(
                f"{backend}/awex/debug/get_parameters",
                json={
                    "save_path": str(saved),
                    "names": list(PROBE_PARAMETERS),
                },
            )
            response.raise_for_status()
            if response.json().get("status") != "ok":
                raise RuntimeError("Could not inspect loaded SGLang parameters")
            actual = torch.load(saved, weights_only=True, map_location="cpu")
            shards = {}
            if getattr(getattr(self.service, "config", None), "tp", 1) > 1:
                from areal.infra.rpc.serialization import deserialize_value

                response = client.post(f"{backend}/awex/report_weight_meta")
                response.raise_for_status()
                metadata = deserialize_value(response.json()["meta"])
                shards = {
                    p.name: shard
                    for p in metadata
                    for shard in p.shards
                    if shard.tp_rank == 0 and p.name in PROBE_PARAMETERS
                }
                if set(shards) != set(PROBE_PARAMETERS):
                    raise RuntimeError("Missing TP shard metadata for weight probes")
            weights_root = (
                checkpoint / "hf" if (checkpoint / "hf").is_dir() else checkpoint
            )
            index_path = weights_root / "model.safetensors.index.json"
            index = (
                json.loads(index_path.read_text())["weight_map"]
                if index_path.exists()
                else {}
            )
            for name in PROBE_PARAMETERS:
                with safe_open(
                    weights_root / index.get(name, "model.safetensors"),
                    framework="pt",
                    device="cpu",
                ) as weights:
                    expected = weights.get_tensor(name)
                if name in shards:
                    shard = shards[name]
                    expected = expected[
                        tuple(
                            slice(offset, offset + length)
                            for offset, length in zip(
                                shard.global_offset, shard.shape, strict=True
                            )
                        )
                    ]
                torch.testing.assert_close(
                    actual[name], expected.to(actual[name].dtype), rtol=0, atol=0
                )
        finally:
            saved.unlink(missing_ok=True)
