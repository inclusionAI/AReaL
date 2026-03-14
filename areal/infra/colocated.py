"""Colocated (GPU time-sharing) orchestration for on-policy training.

In colocated mode, the training engine and inference engine share the same
GPUs and alternate between offloaded/onloaded states. Weights are transferred
through a local disk path (typically ``/dev/shm``) for fast in-memory
synchronization.

Lifecycle per training step::

    [Inference on GPU] -> rollout
    -> offload inference / onload training
    -> train step + save weights to disk
    -> save HF checkpoint + recover checkpoint (train engine still on GPU)
    -> offload training / onload inference + load weights from disk
    -> [Inference on GPU] -> next rollout
"""

from __future__ import annotations

import asyncio
import os
import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch.distributed as dist

import aiohttp
import uvloop

from areal.api.io_struct import WeightUpdateMeta, WeightUpdateRequests
from areal.infra.utils.http import arequest_with_retry, get_default_connector
from areal.utils import logging

if TYPE_CHECKING:
    from areal.api import InferenceEngine, TrainEngine

logger = logging.getLogger("Colocated")


@dataclass
class ColocatedConfig:
    """Configuration for colocated GPU time-sharing.

    Parameters
    ----------
    weight_path : str
        Base path for temporary weight storage during colocated training.
        Defaults to ``/dev/shm/areal_colocated_weights`` for fast in-memory
        transfer. The path should have enough space for the full model weights
        (approximately ``model_size_bytes * 2`` for bf16).
    cleanup_weights_after_load : bool
        Whether to clean up temporary weight files after loading into the
        inference engine. Defaults to True to reclaim ``/dev/shm`` space.
    """

    weight_path: str = "/dev/shm/areal_colocated_weights"
    cleanup_weights_after_load: bool = True


class ColocatedOrchestrator:
    """Orchestrates GPU time-sharing between training and inference engines.

    The orchestrator manages the lifecycle switching between training and
    inference engines that share the same GPUs. It ensures:

    1. Only one engine occupies GPU memory at a time.
    2. Weight transfer happens via local disk (e.g. ``/dev/shm``).
    3. State transitions are idempotent (safe to call multiple times).

    Parameters
    ----------
    train_engine : TrainEngine
        The training engine (FSDP, Megatron, or Archon).
    inf_engine : InferenceEngine
        The inference engine (SGLang or vLLM remote engine).
    config : ColocatedConfig
        Configuration for colocated mode.

    Notes
    -----
    The orchestrator starts with both engines on GPU (``_inf_on_gpu=True``,
    ``_train_on_gpu=True``).  The caller **must** call
    ``initial_offload_training()`` before the first rollout to move
    the training engine off GPU.
    """

    def __init__(
        self,
        train_engine: TrainEngine,
        inf_engine: InferenceEngine,
        config: ColocatedConfig,
    ) -> None:
        self._train_engine: TrainEngine = train_engine
        self._inf_engine: InferenceEngine = inf_engine
        self.config: ColocatedConfig = config

        # Initial state: both engines are on GPU after initialization.
        # The caller MUST call ``initial_offload_training()`` before the
        # first rollout to free GPU memory for inference.
        self._train_on_gpu: bool = True
        self._inf_on_gpu: bool = True

    def initial_offload_training(self) -> None:
        """Offload the training engine after initialization.

        After both engines are initialized, the training engine sits on GPU.
        This method offloads it so the inference engine has exclusive GPU
        access for the first rollout.

        Must be called exactly once, before the first ``prepare_for_training``
        / ``prepare_for_inference`` cycle.
        """
        if not self._train_on_gpu:
            logger.warning(
                "initial_offload_training called but training engine "
                "is not on GPU — skipping."
            )
            return
        logger.info("Initial offload: moving training engine off GPU")
        self._train_engine.offload()
        self._train_on_gpu = False

    def prepare_for_training(self) -> None:
        """Switch GPU ownership from inference to training.

        Offloads the inference engine and onloads the training engine.
        This method is idempotent -- calling it when training is already
        on GPU is a no-op.
        """
        if self._train_on_gpu:
            logger.debug(
                "Training engine already on GPU, skipping prepare_for_training"
            )
            return

        logger.info(
            "Switching to training mode: offloading inference, onloading training"
        )

        # Step 1: offload inference engine (release GPU memory)
        if self._inf_on_gpu:
            if dist.get_rank() == 0:
                self._inf_engine.offload()
            dist.barrier(self._train_engine.cpu_group)
            self._inf_on_gpu = False

        # Step 2: onload training engine (reclaim GPU memory)
        self._train_engine.onload()
        self._train_on_gpu = True

    def prepare_for_inference(self, meta: WeightUpdateMeta) -> None:
        """Switch GPU ownership from training to inference and load new weights.

        Offloads the training engine, onloads the inference engine, and
        triggers a disk-based weight update so the inference engine uses
        the latest trained weights.

        Parameters
        ----------
        meta : WeightUpdateMeta
            Versioned weight update metadata with the disk path where
            the training engine has already saved the latest weights.
        """
        if self._inf_on_gpu:
            logger.debug(
                "Inference engine already on GPU, skipping prepare_for_inference"
            )
            return

        logger.info(
            "Switching to inference mode: offloading training, onloading inference"
        )

        # Step 1: offload training engine (release GPU memory)
        if self._train_on_gpu:
            self._train_engine.offload()
            self._train_on_gpu = False

        # Step 2: onload inference engine (reclaim GPU memory)
        if dist.get_rank() == 0:
            self._inf_engine.onload()
        dist.barrier(self._train_engine.cpu_group)
        self._inf_on_gpu = True

        # Step 3: load new weights from disk into the inference engine
        self._direct_disk_weight_update(meta)

    def sync_weights_to_inference(self, meta: WeightUpdateMeta) -> None:
        """Temporarily onload training, save weights, offload, and sync to inference.

        This method handles the full cycle needed when the training engine's
        weights have been modified while offloaded (e.g. during checkpoint
        recovery) and the inference engine must be brought up-to-date.

        After this call, the training engine is offloaded and the inference
        engine holds the updated weights on GPU.

        Parameters
        ----------
        meta : WeightUpdateMeta
            Versioned weight update metadata.  ``meta.path`` is where the
            HF-format weights will be (temporarily) saved.
        """
        # 1) Onload training engine so we can save weights
        if not self._train_on_gpu:
            self._train_engine.onload()
            self._train_on_gpu = True

        # 2) Caller must save weights to meta.path before calling
        #    prepare_for_inference (which will read from meta.path).
        #    Here we directly switch to inference mode which offloads
        #    training and loads new weights into inference.
        self.prepare_for_inference(meta)

    def direct_disk_weight_update(self, meta: WeightUpdateMeta) -> None:
        """Public wrapper for sending disk weight updates to inference servers.

        Use when the training engine has already saved weights to ``meta.path``
        and the inference engine is already on GPU (e.g. after initial offload).
        """
        self._direct_disk_weight_update(meta)

    def _direct_disk_weight_update(self, meta: WeightUpdateMeta) -> None:
        """Send disk weight update requests directly to inference servers.

        Unlike the standard ``_update_weights_from_disk`` flow in
        ``RemoteInfEngine``, this method bypasses the ``name_resolve.wait()``
        coordination mechanism because in colocated mode:

        1. Training and inference are on the same machine.
        2. The training engine has already finished writing weights before
           this method is called.
        3. No cross-machine synchronization is needed.

        Parameters
        ----------
        meta : WeightUpdateMeta
            Weight update metadata containing the disk path and LoRA config.
        """
        # Access the internal RemoteInfEngine to get backend and addresses.
        # RemoteSGLangEngine / RemotevLLMEngine wraps a RemoteInfEngine
        # via self._engine. We need the backend and addresses from it.
        if dist.get_rank() != 0:
            dist.barrier(self._train_engine.cpu_group)
            return
        engine = self._inf_engine
        inner_engine = getattr(engine, "_engine", engine)
        backend = inner_engine.backend
        addresses: list[str] = inner_engine.addresses
        request_timeout: float = inner_engine.config.request_timeout
        request_retries: int = inner_engine.config.request_retries

        # Build weight update requests using the backend protocol
        weight_reqs: WeightUpdateRequests = (
            backend.build_disk_weight_update_requests(meta)
        )

        logger.info(
            "Sending direct disk weight update to %d server(s) from path: %s",
            len(addresses),
            meta.path,
        )

        async def _send_updates() -> None:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=request_timeout),
                read_bufsize=1024 * 1024 * 10,
                connector=get_default_connector(),
            ) as session:
                for http_req in weight_reqs.requests:
                    jobs = [
                        arequest_with_retry(
                            session=session,
                            addr=addr,
                            endpoint=http_req.endpoint,
                            payload=http_req.payload,
                            method=http_req.method,
                            max_retries=request_retries,
                            timeout=request_timeout,
                        )
                        for addr in addresses
                    ]
                    await asyncio.gather(*jobs)

        uvloop.run(_send_updates())

        logger.info("Direct disk weight update completed")

        # Optionally clean up the weight directory to free /dev/shm space
        if self.config.cleanup_weights_after_load and meta.path is not None:
            shutil.rmtree(meta.path, ignore_errors=True)
            logger.debug("Cleaned up weight directory: %s", meta.path)
        
        dist.barrier(self._train_engine.cpu_group)

    def cleanup(self) -> None:
        """Clean up temporary weight storage.

        Removes the entire weight directory tree. Safe to call even if
        the directory does not exist.
        """
        weight_path = self.config.weight_path
        if os.path.exists(weight_path):
            shutil.rmtree(weight_path, ignore_errors=True)
            logger.info("Cleaned up colocated weight directory: %s", weight_path)
