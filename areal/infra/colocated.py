"""Colocated (GPU time-sharing) orchestration for on-policy training.

In colocated mode, the training engine and inference engine share the same
GPUs and alternate between offloaded/onloaded states.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from areal.api.io_struct import WeightUpdateMeta
from areal.utils import logging

if TYPE_CHECKING:
    from areal.api import InferenceEngine, TrainEngine

logger = logging.getLogger("Colocated")


class ColocatedOrchestrator:
    """Orchestrate GPU ownership between colocated training and inference.
    """

    def __init__(
        self,
        train_engine: TrainEngine,
        inf_engine: InferenceEngine,
    ) -> None:
        self._train_engine: TrainEngine = train_engine
        self._inf_engine: InferenceEngine = inf_engine
        self._train_on_gpu: bool = True
        self._inf_on_gpu: bool = True

    def initial_offload_training(self) -> None:
        """Offload training once so inference owns the GPU before first rollout."""
        if not self._train_on_gpu:
            logger.warning(
                "initial_offload_training called but training engine is already off GPU."
            )
            return

        logger.info("Initial offload: moving training engine off GPU")
        self._train_engine.offload()
        self._train_on_gpu = False

    def prepare_for_training(self) -> None:
        """Switch GPU ownership from inference to training."""
        if self._train_on_gpu:
            logger.debug("Training engine already on GPU, skipping switch")
            return

        logger.info("Switching to training mode")
        if self._inf_on_gpu:
            self._inf_engine.offload()
            self._inf_on_gpu = False

        self._train_engine.onload()
        self._train_on_gpu = True

    def prepare_for_inference(self, meta: WeightUpdateMeta) -> None:
        """Switch GPU ownership from training to inference and sync weights."""
        if self._inf_on_gpu:
            logger.debug("Inference engine already on GPU, skipping switch")
            return

        logger.info("Switching to inference mode")
        if self._train_on_gpu:
            self._train_engine.offload()
            self._train_on_gpu = False

        self._inf_engine.onload()
        self._inf_on_gpu = True
        self._inf_engine.sync_weights_from_disk(meta)
