import json
from typing import Any

import requests

from areal.api.cli_args import (
    InferenceEngineConfig,
    NameResolveConfig,
    TrainEngineConfig,
)
from areal.scheduler.rpc.api import (
    CallEnginePayload,
    ConfigurePayload,
    CreateEnginePayload,
    EngineNameEnum,
    Response,
)
from areal.utils import logging

logger = logging.getLogger(__name__)


class EngineRPCClient:
    def __init__(self, host: str = "localhost", port: int = 5000):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}/api"
        self._request_id = 0

    def _send_request(self, method: str, params: dict[str, Any]) -> Response:
        """Send a JSON-RPC request and parse the standard Response envelope."""

        self._request_id += 1
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": self._request_id,
        }
        headers = {"Content-Type": "application/json"}
        try:
            response = requests.post(
                self.base_url, data=json.dumps(payload), headers=headers, timeout=300
            )
            response.raise_for_status()
            result = response.json()
            if "error" in result:
                raise RuntimeError(f"JSON-RPC error: {result['error']}")
            response_data = result["result"]
            return Response(
                success=response_data["success"],
                message=response_data["message"],
                data=response_data.get("data"),
            )
        except Exception as exc:  # pragma: no cover - network failure path
            logger.error(f"Request failed: {exc}")
            raise

    def create_engine(
        self,
        config: TrainEngineConfig | InferenceEngineConfig,
        class_name: EngineNameEnum,
        initial_args: dict[str, Any],
    ) -> None:
        """Create a remote engine instance.

        This mirrors the payload structure used in test_rpc_integration and
        areal.scheduler.rpc.server.create_app.
        """

        payload = CreateEnginePayload(
            config=config,
            class_name=class_name,
            initial_args=initial_args,
        )
        response = self._send_request(
            "areal.create_engine",
            {"payload": payload.model_dump()},
        )
        if not response.success:
            raise RuntimeError(f"Failed to create engine: {response.message}")

    def call_engine(self, method: str, *args, **kwargs) -> Any:
        """Call a method on the remote engine instance."""

        payload = CallEnginePayload(
            method=method,
            args=list(args),
            kwargs=kwargs,
        )
        response = self._send_request(
            "areal.call_engine",
            {"payload": payload.model_dump()},
        )
        if not response.success:
            raise RuntimeError(f"Failed to call engine: {response.message}")
        return response.data

    def configure(
        self,
        seed_cfg: dict[str, Any] | None = None,
        name_resolve: NameResolveConfig | None = None,
    ) -> None:
        """Configure global settings such as random seed and name_resolve."""

        payload = ConfigurePayload(
            seed_cfg=seed_cfg or {},
            name_resolve=name_resolve or NameResolveConfig(),
        )
        response = self._send_request(
            "areal.configure",
            {"payload": payload.model_dump()},
        )
        if not response.success:
            raise RuntimeError(f"Failed to configure: {response.message}")

    def health(self) -> Response:
        """Health check for the remote engine server."""

        return self._send_request("areal.health", {})

    def export_stats(self, reset: bool = True) -> Any:
        """Export statistics from the remote engine server."""

        response = self._send_request("areal.export_stats", {"reset": reset})
        if not response.success:
            raise RuntimeError(f"Failed to export stats: {response.message}")
        return response.data
