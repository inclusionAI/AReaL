from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any

import requests

from areal.api import ParamSpec, WeightUpdateMeta
from areal.infra.rpc.serialization import serialize_value
from areal.infra.utils.concurrent import get_executor
from areal.utils import logging

logger = logging.getLogger(__name__)


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

    # Bypass HTTP_PROXY / HTTPS_PROXY for all internal callback requests.
    # In environments where a corporate HTTP proxy is configured (e.g.,
    # HTTP_PROXY=http://sys-proxy-rd-relay.byted.org:8118), Python's
    # requests library auto-routes through the proxy. When the callback
    # server's address (e.g., an IPv6 pod IP) is not listed in NO_PROXY,
    # the proxy intercepts the request and applies its own timeout (~60s),
    # causing 504 Gateway Timeout on long-running NCCL weight updates
    # before the operation completes. Using a Session with trust_env=False
    # ensures direct connection to the callback server.
    _no_proxy_session: requests.Session | None = field(default=None, init=False, repr=False)

    @property
    def _session(self) -> requests.Session:
        """Lazily create a requests.Session that bypasses env proxy settings."""
        if self._no_proxy_session is None:
            s = requests.Session()
            s.trust_env = False  # Ignore HTTP_PROXY / HTTPS_PROXY / NO_PROXY
            object.__setattr__(self, "_no_proxy_session", s)
        return self._no_proxy_session

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
            resp = self._session.post(
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
        return get_executor().submit(self._post, endpoint, payload)

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

    def update_weights_from_tensor(
        self,
        named_tensors: list | None = None,
        tp_size: int = 1,
        flush_cache: bool = True,
        serialized_payload: dict | None = None,
    ) -> None:
        """Callback to controller to update EAGLE draft model MTP weights.

        In single-controller mode, tensor data is pre-serialized by the
        MegatronEngine before calling this method. The serialized payload
        (base64-encoded CUDA IPC handles) travels through the HTTP callback
        chain to the RolloutController, which delegates to the inference
        engine workers.

        This is synchronous (blocking) because the MTP tensor update
        happens AFTER the main NCCL weight sync (within the pause window),
        so there is no deadlock risk from blocking.

        Parameters
        ----------
        named_tensors : list, optional
            Ignored in callback mode — tensors must be pre-serialized.
        tp_size : int
            Ignored in callback mode — encoded in serialized_payload.
        flush_cache : bool
            Whether to flush KV cache after update.
        serialized_payload : dict, optional
            Pre-serialized payload for /update_weights_from_tensor endpoint.
        """
        if serialized_payload is None:
            raise ValueError(
                "RolloutCallback.update_weights_from_tensor requires "
                "serialized_payload (pre-serialized tensor data). "
                "Raw tensor mode is not supported through the callback chain."
            )
        payload = {
            "serialized_payload": serialize_value(serialized_payload),
        }
        # Use non-blocking POST as defense-in-depth. Even with proxy bypass,
        # large MTP tensor payloads may take time to transmit. The fire-and-
        # forget callback pattern ensures the HTTP layer does not block.
        fut = self._post_nowait_void("/callback/update_weights_tensor", payload)
        try:
            fut.result(timeout=120)
        except TimeoutError:
            logger.warning(
                "update_weights_from_tensor callback timed out. "
                "Tensor update dispatched via fire-and-forget."
            )
        except Exception as e:
            logger.warning(
                f"update_weights_from_tensor callback error: {e}. "
                "Tensor update dispatched via fire-and-forget."
            )
