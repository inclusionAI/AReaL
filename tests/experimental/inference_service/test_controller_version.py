"""Unit tests for GatewayInferenceController version management.

Tests set_version, get_version, and get_worker_versions with mocked HTTP calls.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from areal.api.cli_args import SchedulingSpec
from areal.experimental.inference_service.controller.config import (
    GatewayControllerConfig,
)
from areal.experimental.inference_service.controller.controller import (
    GatewayInferenceController,
)


# =============================================================================
# Helpers
# =============================================================================


def _make_controller(
    gateway_addr: str = "",
    worker_ids: dict[str, str] | None = None,
    version: int = 0,
) -> GatewayInferenceController:
    """Create a controller with minimal config and manually injected state.

    Does NOT call initialize() — internal fields are set directly.
    """
    cfg = GatewayControllerConfig(
        scheduling_spec=(SchedulingSpec(),),
    )
    scheduler = MagicMock()
    ctrl = GatewayInferenceController(config=cfg, scheduler=scheduler)
    ctrl._gateway_addr = gateway_addr
    ctrl._worker_ids = worker_ids or {}
    ctrl._version = version
    return ctrl


# =============================================================================
# TestControllerSetVersion
# =============================================================================


class TestControllerSetVersion:
    """Test GatewayInferenceController.set_version."""

    def test_set_version_updates_local(self):
        ctrl = _make_controller()
        ctrl.set_version(5)
        assert ctrl._version == 5

    def test_set_version_no_gateway_skips_broadcast(self):
        """When _gateway_addr is empty, set_version updates local but makes no HTTP calls."""
        ctrl = _make_controller(gateway_addr="", worker_ids={"dp0": "w1"})
        with patch.object(
            ctrl, "_async_gateway_http_post", new_callable=AsyncMock
        ) as mock_post:
            ctrl.set_version(5)
            mock_post.assert_not_called()
        assert ctrl._version == 5

    def test_set_version_broadcasts_to_all_workers(self):
        """When gateway_addr is set and 2 workers exist, broadcasts to both."""
        ctrl = _make_controller(
            gateway_addr="http://gateway:8000",
            worker_ids={"dp0": "w1", "dp1": "w2"},
        )

        # Patch _async_gateway_http_post to be a no-op async mock
        mock_post = AsyncMock()

        # Patch run_async_task to directly call the async function synchronously
        # via an event loop, so we can verify the calls to _async_gateway_http_post.
        import asyncio

        def _run_async_task_sync(coro_fn, *args, **kwargs):
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(coro_fn(*args, **kwargs))
            finally:
                loop.close()

        with patch.object(ctrl, "_async_gateway_http_post", mock_post):
            with patch(
                "areal.experimental.inference_service.controller.controller.GatewayInferenceController.set_version",
                wraps=None,
            ):
                # Directly call the internal async method to verify broadcast
                asyncio.get_event_loop_policy()
                loop = asyncio.new_event_loop()
                try:
                    loop.run_until_complete(ctrl._async_set_version(10))
                finally:
                    loop.close()

        assert mock_post.call_count == 2
        call_endpoints = [call.args[0] for call in mock_post.call_args_list]
        assert "/set_version/w1" in call_endpoints
        assert "/set_version/w2" in call_endpoints


# =============================================================================
# TestControllerGetVersion
# =============================================================================


class TestControllerGetVersion:
    """Test GatewayInferenceController.get_version."""

    def test_get_version_returns_local(self):
        ctrl = _make_controller(version=0)
        assert ctrl.get_version() == 0

    def test_get_version_after_set(self):
        ctrl = _make_controller()
        ctrl.set_version(7)
        assert ctrl.get_version() == 7


# =============================================================================
# TestControllerGetWorkerVersions
# =============================================================================


class TestControllerGetWorkerVersions:
    """Test GatewayInferenceController.get_worker_versions."""

    def test_get_worker_versions_single_worker(self):
        """Mock _async_get_worker_versions to return a single worker's version."""
        ctrl = _make_controller(
            gateway_addr="http://gateway:8000",
            worker_ids={"dp0": "w1"},
            version=0,
        )

        async def _mock_get_versions(worker_id=None):
            return 5

        with patch(
            "areal.infra.utils.concurrent.run_async_task",
            side_effect=lambda fn, *a, **kw: 5,
        ):
            result = ctrl.get_worker_versions(worker_id="w1")
        assert result == 5

    def test_get_worker_versions_all_same(self):
        """2 workers both return version 5 → returns int 5."""
        ctrl = _make_controller(
            gateway_addr="http://gateway:8000",
            worker_ids={"dp0": "w1", "dp1": "w2"},
            version=0,
        )

        with patch(
            "areal.infra.utils.concurrent.run_async_task",
            side_effect=lambda fn, *a, **kw: 5,
        ):
            result = ctrl.get_worker_versions()
        assert result == 5

    def test_get_worker_versions_different(self):
        """2 workers return different versions → returns dict."""
        ctrl = _make_controller(
            gateway_addr="http://gateway:8000",
            worker_ids={"dp0": "w1", "dp1": "w2"},
            version=0,
        )

        expected = {"w1": 3, "w2": 5}
        with patch(
            "areal.infra.utils.concurrent.run_async_task",
            side_effect=lambda fn, *a, **kw: expected,
        ):
            result = ctrl.get_worker_versions()
        assert result == {"w1": 3, "w2": 5}

    def test_get_worker_versions_no_gateway_returns_local(self):
        """When _gateway_addr is empty, falls back to local _version."""
        ctrl = _make_controller(gateway_addr="", version=42)
        result = ctrl.get_worker_versions()
        assert result == 42
