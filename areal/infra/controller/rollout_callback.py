import sys
import time
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any

import requests

from areal.api import ParamSpec, WeightUpdateMeta
from areal.infra.rpc.serialization import serialize_value
from areal.infra.utils.concurrent import get_executor
from areal.utils import logging

logger = logging.getLogger(__name__)

# Direct-connect proxy setting: tells requests to bypass all env proxies.
# This is used for every callback HTTP request to avoid corporate proxy
# interference (504 Gateway Timeout) on internal pod-to-pod communication.
_NO_PROXY = {"http": None, "https": None}


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

        Uses ``proxies=_NO_PROXY`` to bypass environment proxy variables
        (HTTP_PROXY / HTTPS_PROXY / NO_PROXY).  In environments where a
        corporate HTTP proxy is configured (e.g.,
        HTTP_PROXY=http://sys-proxy-rd-relay.byted.org:8118), Python's
        ``requests`` library auto-routes through the proxy.  When the
        callback server address (e.g., an IPv6 pod IP) is not listed in
        NO_PROXY, the proxy intercepts the request and applies its own
        timeout (~60 s), causing 504 Gateway Timeout on long-running NCCL
        weight updates.  Passing ``proxies={"http": None, "https": None}``
        ensures a direct connection to the callback server on every call,
        with zero extra state — which is critical because this dataclass is
        serialized across RPC boundaries by AReaL's ``serialize_value`` /
        ``deserialize_value`` (adding non-init fields would break
        deserialization).

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
        _t_post = time.time()
        logger.info(
            f"[DiagMTP][Callback] _post: sending POST to {url} "
            f"(timeout={self.request_timeout}s, proxies={_NO_PROXY})"
        )
        try:
            resp = requests.post(
                url,
                json=payload or {},
                timeout=self.request_timeout,
                proxies=_NO_PROXY,
            )
            _elapsed = time.time() - _t_post
            logger.info(
                f"[DiagMTP][Callback] _post: response received from {url} "
                f"in {_elapsed:.3f}s (status={resp.status_code})"
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            _elapsed = time.time() - _t_post
            logger.error(
                f"[DiagMTP][Callback] _post: FAILED {url} after {_elapsed:.3f}s: {e}"
            )
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
        _t0 = time.time()
        _n_snt = (
            len(serialized_payload.get("serialized_named_tensors", []))
            if isinstance(serialized_payload, dict)
            else 0
        )
        _snt_b64_len = (
            sum(len(s) for s in serialized_payload.get("serialized_named_tensors", []))
            if isinstance(serialized_payload, dict)
            else 0
        )
        logger.info(
            f"[DiagMTP][Callback] update_weights_from_tensor ENTERED. "
            f"serialized_payload keys={list(serialized_payload.keys())}, "
            f"n_serialized_tensors={_n_snt}, "
            f"total_b64_bytes={_snt_b64_len} ({_snt_b64_len / 1024 / 1024:.2f} MB), "
            f"controller_addr={self.controller_addr}"
        )
        payload = {
            "serialized_payload": serialize_value(serialized_payload),
        }
        _payload_size = sys.getsizeof(str(payload))
        _t1 = time.time()
        logger.info(
            f"[DiagMTP][Callback] serialize_value took {_t1 - _t0:.3f}s, "
            f"payload_approx_size={_payload_size} bytes "
            f"({_payload_size / 1024 / 1024:.2f} MB)"
        )

        # Synchronous blocking POST: MTP tensor update must complete before
        # training proceeds to continue_generation. This follows verl/slime's
        # approach of fully blocking weight updates. The callback server
        # handler is also blocking (.result()), so HTTP 200 guarantees the
        # tensor update is finished on the inference side.
        logger.info(
            f"[DiagMTP][Callback] Calling _post('/callback/update_weights_tensor') "
            f"with timeout={self.request_timeout}s..."
        )
        try:
            self._post("/callback/update_weights_tensor", payload)
            _t2 = time.time()
            logger.info(
                f"[DiagMTP][Callback] _post completed in {_t2 - _t1:.3f}s "
                f"(total: {_t2 - _t0:.3f}s)"
            )
        except Exception as e:
            _t2 = time.time()
            logger.error(
                f"[DiagMTP][Callback] _post FAILED after {_t2 - _t1:.3f}s "
                f"(total: {_t2 - _t0:.3f}s): {type(e).__name__}: {e}"
            )
            raise
