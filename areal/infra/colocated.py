"""Colocated (GPU time-sharing) orchestration for on-policy training.

In colocated mode, the training engine and inference engine share the same
GPUs and alternate between offloaded/onloaded states.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch.distributed as dist

from areal.api.io_struct import WeightUpdateMeta
from areal.utils import logging

if TYPE_CHECKING:
    from areal.api import InferenceEngine, TrainEngine

logger = logging.getLogger("Colocated")


class ColocatedOrchestrator:
    """Orchestrate GPU ownership between colocated training and inference."""

    def __init__(
        self,
        train_engine: TrainEngine,
        inf_engine: InferenceEngine,
    ) -> None:
        self._train_engine: TrainEngine = train_engine
        self._inf_engine: InferenceEngine = inf_engine
        self._train_on_gpu: bool = True
        self._inf_on_gpu: bool = True

    def _is_rollout_coordinator(self) -> bool:
        return not dist.is_initialized() or dist.get_rank() == 0

    def _barrier(self) -> None:
        if not dist.is_initialized():
            return
        cpu_group = self._train_engine.cpu_group
        if cpu_group is None:
            dist.barrier()
            return
        dist.barrier(group=cpu_group)

    def initial_offload_training(self) -> None:
        """Offload training once so inference owns the GPU before first rollout."""
        if not self._train_on_gpu:
            logger.warning(
                "initial_offload_training called but training engine is already off GPU."
            )
            return

        if self._is_rollout_coordinator():
            logger.info("Initial offload: moving training engine off GPU")
        self._train_engine.offload()
        self._train_on_gpu = False

    def prepare_for_training(self) -> None:
        """Switch GPU ownership from inference to training."""
        if self._train_on_gpu:
            logger.debug("Training engine already on GPU, skipping switch")
            return

        if self._is_rollout_coordinator():
            logger.info("Switching to training mode")

        # Pause local submission on every rank first so no new requests are queued.
        self._inf_engine.pause()

        # Only one coordinator should touch the shared rollout servers.
        if self._is_rollout_coordinator():
            self._inf_engine.pause_generation()
            if self._inf_on_gpu:
                self._inf_engine.offload()

        self._barrier()
        self._inf_on_gpu = False

        # All training ranks must participate in the training-engine collective.
        self._train_engine.onload()
        self._train_on_gpu = True

    def prepare_for_inference(self, meta: WeightUpdateMeta) -> None:
        """Switch GPU ownership from training to inference and sync weights."""
        if self._inf_on_gpu:
            logger.debug("Inference engine already on GPU, skipping switch")
            return

        if self._is_rollout_coordinator():
            logger.info("Switching to inference mode")

        if self._train_on_gpu:
            self._train_engine.offload()
        self._train_on_gpu = False

        self._barrier()

        if self._is_rollout_coordinator():
            self._inf_engine.onload()

            if meta.version is None:
                raise ValueError("Colocated disk weight sync requires meta.version.")

            # Colocated flow publishes the ready signal before trainer later calls
            # rollout.set_version(new_version), so align the rollout-side wait key here.
            self._inf_engine.set_version(meta.version)
            self._inf_engine.sync_weights_from_disk(meta)
            self._inf_engine.continue_generation()

        self._barrier()
        self._inf_engine.resume()
        self._inf_on_gpu = True
