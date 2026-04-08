from concurrent.futures import Future
from dataclasses import dataclass
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

    IMPORTANT: Methods that involve NCCL collective operations MUST be non-blocking
    (return Future). NCCL operations are collective - both train and inference sides
    must participate concurrently. If these methods blocked, the train side couldn't
    start its NCCL operations while waiting for the inference side, causing a deadlock.
    """

    controller_addr: str
    request_timeout: float = 600.0

    def _post(self, endpoint: str, payload: dict[str, Any] | None = None) -> dict:
        url = f"http://{self.controller_addr}{endpoint}"
        try:
            logger.info(
                "[RolloutCallback] POST %s  timeout=%.1fs  payload_keys=%s",
                url,
                self.request_timeout,
                list((payload or {}).keys()),
            )
            import time as _time

            _t0 = _time.monotonic()
            resp = requests.post(
                url,
                json=payload or {},
                timeout=self.request_timeout,
            )
            _elapsed = _time.monotonic() - _t0
            resp.raise_for_status()
            logger.info(
                "[RolloutCallback] POST %s completed in %.2fs  status=%d",
                endpoint,
                _elapsed,
                resp.status_code,
            )
            return resp.json()
        except requests.exceptions.Timeout:
            logger.error(
                "[RolloutCallback] TIMEOUT on POST %s after %.1fs. "
                "This usually indicates NCCL group init or weight broadcast is hanging. "
                "Check SGLang worker logs for rank collision or NCCL errors.",
                url,
                self.request_timeout,
            )
            raise
        except Exception as e:
            logger.error("[RolloutCallback] POST %s FAILED: %s", url, repr(e))
            raise

    def _post_nowait(
        self, endpoint: str, payload: dict[str, Any] | None = None
    ) -> Future[dict]:
        logger.info(
            "[RolloutCallback] Submitting async POST %s (non-blocking for NCCL collective)",
            endpoint,
        )
        return get_executor().submit(self._post, endpoint, payload)

    def _post_nowait_void(
        self, endpoint: str, payload: dict[str, Any] | None = None
    ) -> Future[None]:
        def _fn():
            self._post(endpoint, payload)

        return get_executor().submit(_fn)

    def pause_generation(self) -> dict:
        logger.info("[RolloutCallback] >>> pause_generation")
        return self._post("/callback/pause_generation")

    def continue_generation(self) -> dict:
        logger.info("[RolloutCallback] >>> continue_generation")
        return self._post("/callback/continue_generation")

    def init_weights_update_group(self, meta: WeightUpdateMeta) -> Future[dict]:
        """Initialize the NCCL weight-update process group on the rollout side.

        MUST be non-blocking (returns Future). The calling code in
        megatron_engine._init_weight_update_from_distributed() does:

            fut = self.rollout_engine.init_weights_update_group(meta)  # non-blocking
            init_custom_process_group(rank=0, ...)   # Megatron joins as rank 0
            fut.result()                             # wait for rollout side

        If this were synchronous, it would deadlock: Megatron blocks waiting for
        the HTTP response, but the SGLang workers block in init_custom_process_group
        waiting for rank 0 (Megatron) to join, which can never happen.
        """
        payload = {"meta": serialize_value(meta)}
        logger.info(
            "[RolloutCallback] >>> init_weights_update_group (async)  "
            "nccl_master=%s:%s  group=%s  world_size=%s",
            getattr(meta, "nccl_master_address", "?"),
            getattr(meta, "nccl_master_port", "?"),
            getattr(meta, "nccl_group_name", "?"),
            getattr(getattr(meta, "gen_allocation", None), "parallel", None)
            and meta.gen_allocation.parallel.world_size + 1,
        )
        return self._post_nowait("/callback/init_weights_group", payload)

    def update_weights_from_distributed(
        self, meta: WeightUpdateMeta, param_specs: list[ParamSpec]
    ) -> Future[None]:
        """Update weights via NCCL broadcast. Must be non-blocking (returns Future)."""
        payload = {
            "meta": serialize_value(meta),
            "param_specs": serialize_value(param_specs),
        }
        logger.info(
            "[RolloutCallback] >>> update_weights_from_distributed (async)  "
            "group=%s  n_params=%d",
            getattr(meta, "nccl_group_name", "?"),
            len(param_specs),
        )
        return self._post_nowait_void("/callback/update_weights_xccl", payload)

    def update_weights_from_disk(self, meta: WeightUpdateMeta) -> Future[None]:
        payload = {"meta": serialize_value(meta)}
        logger.info("[RolloutCallback] >>> update_weights_from_disk (async)")
        return self._post_nowait_void("/callback/update_weights_disk", payload)

    def set_version(self, version: int) -> dict:
        payload = {"version": version}
        logger.info("[RolloutCallback] >>> set_version(%d)", version)
        return self._post("/callback/set_version", payload)