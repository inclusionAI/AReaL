import atexit
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import requests

from areal.api.io_struct import ParamSpec, WeightUpdateMeta
from areal.scheduler.rpc.serialization import serialize_value
from areal.utils import logging

logger = logging.getLogger(__name__)

# Lazy-initialized thread pool for async HTTP requests
_executor: ThreadPoolExecutor | None = None
_executor_lock = threading.Lock()


def _get_executor() -> ThreadPoolExecutor:
    """Get or create the shared thread pool executor."""
    global _executor
    if _executor is None:
        with _executor_lock:
            if _executor is None:
                _executor = ThreadPoolExecutor(
                    max_workers=4, thread_name_prefix="rollout_callback"
                )
                # Register cleanup on process exit
                atexit.register(_shutdown_executor)
    return _executor


def _shutdown_executor() -> None:
    """Shutdown the shared thread pool executor if it exists.

    Called via atexit at process exit, when no other threads should be
    accessing the executor.
    """
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=False)
        _executor = None


@dataclass
class RolloutCallback:
    """Callback interface for train workers to coordinate with TrainController.

    This class acts as a proxy that train engines use to trigger operations on
    the inference side via HTTP callbacks to the TrainController. The controller
    then forwards these to the RolloutController.

    IMPORTANT: Methods that return Future must be non-blocking to avoid deadlocks.
    NCCL operations are collective - both train and inference sides must participate
    concurrently. If these methods blocked, the train side couldn't start its NCCL
    operations while waiting for the inference side, causing a deadlock.
    """

    controller_addr: str
    request_timeout: float = 600.0

    def _post(self, endpoint: str, payload: dict[str, Any] | None = None) -> dict:
        """Make synchronous HTTP POST to controller callback endpoint.

        Parameters
        ----------
        endpoint : str
            The callback endpoint (e.g., "/callback/init_weights_group")
        payload : dict, optional
            JSON payload to send

        Returns
        -------
        dict
            Response JSON from controller
        """
        url = f"http://{self.controller_addr}{endpoint}"
        try:
            resp = requests.post(
                url,
                json=payload or {},
                timeout=self.request_timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.error(f"Callback to {url} failed: {e}")
            raise

    def _post_nowait(
        self, endpoint: str, payload: dict[str, Any] | None = None
    ) -> Future[dict]:
        """Make asynchronous HTTP POST to controller callback endpoint.

        This method submits the HTTP request to a background thread and returns
        immediately with a Future. This is critical for NCCL coordination where
        both train and inference sides must participate in collective operations
        concurrently.

        Parameters
        ----------
        endpoint : str
            The callback endpoint
        payload : dict, optional
            JSON payload to send

        Returns
        -------
        Future[dict]
            Future that completes when the HTTP response is received
        """
        return _get_executor().submit(self._post, endpoint, payload)

    def _post_nowait_void(
        self, endpoint: str, payload: dict[str, Any] | None = None
    ) -> Future[None]:
        """Make an async POST request and return a Future that resolves to None."""
        http_future = self._post_nowait(endpoint, payload)
        result_future: Future[None] = Future()

        def on_done(f: Future[dict]):
            try:
                f.result()  # Raise any exception from the HTTP request
                result_future.set_result(None)
            except Exception as e:
                result_future.set_exception(e)

        http_future.add_done_callback(on_done)
        return result_future

    def init_weights_update_group(self, meta: WeightUpdateMeta) -> Future[None]:
        """Callback to controller to initialize weight update group on inference side.

        This method is NON-BLOCKING. It starts the HTTP request in a background
        thread and returns immediately. This allows the train engine to proceed
        with creating its side of the NCCL group concurrently.

        Parameters
        ----------
        meta : WeightUpdateMeta
            Weight update metadata

        Returns
        -------
        Future[None]
            Future that completes when controller finishes initialization
        """
        payload = {"meta": serialize_value(meta)}
        return self._post_nowait_void("/callback/init_weights_group", payload)

    def update_weights_from_distributed(
        self, meta: WeightUpdateMeta, param_specs: list[ParamSpec]
    ) -> Future[None]:
        """Callback to controller to receive weights on inference side.

        This method is NON-BLOCKING. The train engine calls this to notify the
        inference side to start receiving NCCL broadcasts, then immediately
        starts broadcasting. Both sides participate in the NCCL collective
        concurrently.

        Parameters
        ----------
        meta : WeightUpdateMeta
            Weight update metadata
        param_specs : list[ParamSpec]
            List of parameter specifications for this update batch

        Returns
        -------
        Future[None]
            Future that completes when controller finishes receiving weights
        """
        payload = {
            "meta": serialize_value(meta),
            "param_specs": serialize_value(param_specs),
        }
        return self._post_nowait_void("/callback/update_weights_xccl", payload)

    def update_weights_from_disk(self, meta: WeightUpdateMeta) -> Future[None]:
        """Callback to controller to load weights from disk on inference side.

        This method is NON-BLOCKING for consistency, though disk-based updates
        don't have the same NCCL coordination requirements.

        Parameters
        ----------
        meta : WeightUpdateMeta
            Weight update metadata with path information

        Returns
        -------
        Future[None]
            Future that completes when controller finishes loading weights
        """
        payload = {"meta": serialize_value(meta)}
        return self._post_nowait_void("/callback/update_weights_disk", payload)

    def pause_generation(self) -> None:
        """Callback to controller to pause inference generation.

        This is synchronous as it must complete before weight updates begin.
        """
        self._post("/callback/pause_generation")

    def continue_generation(self) -> None:
        """Callback to controller to resume inference generation.

        This is synchronous as it should complete before returning control.
        """
        self._post("/callback/continue_generation")
