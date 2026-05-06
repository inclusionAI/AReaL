# SPDX-License-Identifier: Apache-2.0
"""WeightTransportActor for TW tensor transport via RDT.

Created by TW subprocess, shares same GPU via CUDA_VISIBLE_DEVICES.
Receives sliced tensor IPC handles and implements @ray.method(tensor_transport).
"""

from __future__ import annotations

import threading
from collections import defaultdict
from threading import Condition
from typing import Any

import ray
import torch

from areal.utils import logging

logger = logging.getLogger("WeightTransportActor")


@ray.remote
class WeightTransportActor:
    """Actor for weight tensor transport via RDT.

    Key features:
    - max_concurrency set to IW count (multiple IW may pull concurrently)
    - Condition key: {pair_name}/{infer_rank}/{version} for one-to-one TW-IW sync
    - IW calls clear_ipc_handles() after ray.get() to release shared GPU memory
    """

    def __init__(self):
        # Version-based tensor storage: {pair_name}/{infer_rank}/{version}/{param_name}
        self._tensors: dict[str, torch.Tensor] = {}
        self._tensors_lock = threading.Lock()  # Protect _tensors dict access

        # Synchronization: wait for IPC handles ready
        self._tensor_ready_lock = threading.Lock()
        self._tensor_ready: dict[str, Condition] = {}
        self._tensor_ready_flags: dict[str, bool] = defaultdict(bool)

        # Activate cuda:0 for IPC (CUDA_VISIBLE_DEVICES already set to single GPU by TW)
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            logger.info("WeightTransportActor initialized on cuda:0")

    def store_ipc_handles(
        self,
        pair_name: str,
        infer_rank: int,
        version: int,
        ipc_handles: dict[str, Any],
    ) -> None:
        """Receive all IPC handles, store tensors and notify IW.

        Args:
            pair_name: TW-IW pair identifier
            infer_rank: IW's global rank
            version: Weight version number
            ipc_handles: dict of {param_name: ipc_payload}
        """
        prefix = f"{pair_name}/{infer_rank}/{version}"

        # Build tensors outside lock to reduce lock holding time
        new_tensors: dict[str, torch.Tensor] = {}
        for param_name, ipc_payload in ipc_handles.items():
            rebuild_fn = ipc_payload["rebuild_fn"]
            tensor_meta = ipc_payload["tensor_meta"]
            shared_tensor = rebuild_fn(*tensor_meta)
            key = f"{prefix}/{param_name}"
            new_tensors[key] = shared_tensor

        # Store tensors under lock (brief operation)
        with self._tensors_lock:
            self._tensors.update(new_tensors)

        # Notify waiting IWs
        with self._tensor_ready_lock:
            self._tensor_ready_flags[prefix] = True
            if prefix in self._tensor_ready:
                self._tensor_ready[prefix].notify_all()

        logger.info(f"Stored {len(ipc_handles)} IPC handles for {prefix}")

    def _wait_for_ready(
        self, pair_name: str, infer_rank: int, version: int, timeout: float = 30.0
    ) -> bool:
        """Wait for IPC handles ready (blocking)."""
        prefix = f"{pair_name}/{infer_rank}/{version}"

        with self._tensor_ready_lock:
            if self._tensor_ready_flags.get(prefix, False):
                return True

            if prefix not in self._tensor_ready:
                self._tensor_ready[prefix] = Condition(self._tensor_ready_lock)

            return self._tensor_ready[prefix].wait(timeout=timeout)

    @ray.method(tensor_transport="NIXL")
    def get_weights_tensor_nixl(
        self,
        pair_name: str,
        infer_rank: int,
        version: int,
    ) -> dict[str, torch.Tensor]:
        """Tensor transport for GPU (NIXL backend).

        IW calls this method, blocks until TW stores IPC handles.
        Returns cloned tensors for NIXL RDMA compatibility.
        """
        prefix = f"{pair_name}/{infer_rank}/{version}"

        if not self._wait_for_ready(pair_name, infer_rank, version):
            raise RuntimeError(f"IPC handles not ready for {prefix} after 30s")

        # Clone tensors outside lock to reduce lock holding time
        with self._tensors_lock:
            tensors_to_clone = {
                k: v for k, v in self._tensors.items() if k.startswith(prefix)
            }

        if not tensors_to_clone:
            raise RuntimeError(f"Tensors not found for {prefix}")

        result = {}
        for key, tensor in tensors_to_clone.items():
            param_name = key.split("/")[-1]
            # Clone tensor to create memory in Actor's CUDA context
            # This allows NIXL RDMA registration (IPC shared memory cannot be registered)
            result[param_name] = tensor.clone().detach()

        return result

    @ray.method(tensor_transport="NIXL")
    def warmup_nixl(self) -> dict[str, torch.Tensor]:
        """Warmup NIXL agent by returning a minimal tensor.

        IW calls this during init to trigger NIXL agent initialization
        on both IW (driver) and TW Actor sides before actual weight transfer.
        This moves ~9s initialization overhead from update_weights to connect phase.
        """
        # Create a tiny tensor to trigger NIXL registration
        warmup_tensor = torch.zeros(1, dtype=torch.float32, device="cuda:0")
        logger.info("NIXL warmup tensor created")
        return {"warmup": warmup_tensor}

    # TODO: Implement YR backend for NPU (ray-ascend)
    # @ray.method(tensor_transport="YR")
    # def get_weights_tensor_yr(
    #     self,
    #     pair_name: str,
    #     infer_rank: int,
    #     version: int,
    # ) -> dict[str, torch.Tensor]:
    #     """Tensor transport for NPU (YR backend)."""
    #     ...

    def clear_ipc_handles(self, pair_name: str, infer_rank: int, version: int) -> None:
        """Clean up IPC handles for specific infer_rank and version."""
        prefix = f"{pair_name}/{infer_rank}/{version}"

        # Remove tensors under lock
        with self._tensors_lock:
            for key in list(self._tensors.keys()):
                if key.startswith(prefix):
                    del self._tensors[key]

        # Clear readiness state
        with self._tensor_ready_lock:
            self._tensor_ready_flags[prefix] = False
            if prefix in self._tensor_ready:
                del self._tensor_ready[prefix]
