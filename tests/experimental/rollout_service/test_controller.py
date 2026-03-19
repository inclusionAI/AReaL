"""Tests for GatewayRolloutController."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from areal.experimental.rollout_service.controller.config import GatewayControllerConfig
from areal.experimental.rollout_service.controller.controller import (
    GatewayRolloutController,
)


# =============================================================================
# GatewayControllerConfig
# =============================================================================


class TestGatewayControllerConfig:
    def test_defaults(self):
        cfg = GatewayControllerConfig()
        assert cfg.admin_api_key == "areal-admin-key"
        assert cfg.consumer_batch_size == 16
        assert cfg.max_concurrent_rollouts is None
        assert cfg.max_head_offpolicyness == 0
        assert cfg.enable_rollout_tracing is False

    def test_custom_values(self):
        cfg = GatewayControllerConfig(
            admin_api_key="custom-key",
            consumer_batch_size=32,
            max_concurrent_rollouts=64,
            max_head_offpolicyness=5,
        )
        assert cfg.admin_api_key == "custom-key"
        assert cfg.consumer_batch_size == 32
        assert cfg.max_concurrent_rollouts == 64
        assert cfg.max_head_offpolicyness == 5

    def test_scheduling_fields(self):
        cfg = GatewayControllerConfig(
            request_timeout=60.0,
            setup_timeout=600.0,
        )
        assert cfg.request_timeout == 60.0
        assert cfg.setup_timeout == 600.0

    def test_dump_to_file_defaults_to_false(self):
        cfg = GatewayControllerConfig()
        assert cfg.dump_to_file is False


# =============================================================================
# GatewayRolloutController — workflow resolution helpers
# =============================================================================


class TestControllerWorkflowResolution:
    def test_resolve_workflow_with_instance(self):
        """Test _resolve_workflow raises for non-RolloutWorkflow instances without proxy."""
        # Non-RolloutWorkflow instances hit case 5 (agent-like) and need proxy_addr + controller.
        with pytest.raises(ValueError, match="proxy_addr and controller are required"):
            GatewayRolloutController._resolve_workflow(12345)

    def test_resolve_workflow_none_raises(self):
        with pytest.raises(ValueError, match="must be specified"):
            GatewayRolloutController._resolve_workflow(None)

    def test_resolve_should_accept_fn_none(self):
        assert GatewayRolloutController._resolve_should_accept_fn(None) is None

    def test_resolve_should_accept_fn_callable(self):
        fn = lambda x: True  # noqa: E731
        assert GatewayRolloutController._resolve_should_accept_fn(fn) is fn

    def test_resolve_workflow_with_agent_class(self):
        """Test _resolve_workflow wraps agent-like classes in OpenAIProxyWorkflow."""
        cfg = GatewayControllerConfig()
        scheduler = MagicMock()
        controller = GatewayRolloutController(config=cfg, scheduler=scheduler)
        controller._gateway_addr = "http://test:8080"

        class MockAgent:
            async def run(self, data, **kwargs):
                return 1.0

        resolved = GatewayRolloutController._resolve_workflow(
            MockAgent,
            workflow_kwargs={},
            proxy_addr="http://test:8080",
            controller=controller,
        )
        # Avoid importing areal.experimental.openai at module/test level
        # (PEP 695 syntax in that package breaks Python 3.10-3.12 collection).
        assert type(resolved).__name__ == "OpenAIProxyWorkflow"
        assert hasattr(resolved, "arun_episode")

    def test_resolve_workflow_agent_class_without_proxy_raises(self):
        """Test _resolve_workflow raises when agent class given without proxy_addr."""

        class MockAgent:
            async def run(self, data, **kwargs):
                return 1.0

        with pytest.raises(ValueError, match="proxy_addr and controller are required"):
            GatewayRolloutController._resolve_workflow(MockAgent, workflow_kwargs={})


# =============================================================================
# GatewayRolloutController — API surface
# =============================================================================


class TestGatewayRolloutControllerAPISurface:
    def test_has_all_public_methods(self):
        methods = [
            "initialize",
            "destroy",
            "submit",
            "wait",
            "rollout_batch",
            "prepare_batch",
            "chat_completion",
            "set_version",
            "get_version",
            "get_capacity",
            "pause",
            "resume",
            "export_stats",
            "pause_generation",
            "continue_generation",
            "config_perf_tracer",
            "save_perf_tracer",
            "start_proxy",
            "start_proxy_gateway",
        ]
        for m in methods:
            assert hasattr(GatewayRolloutController, m), f"Missing method: {m}"

    def test_has_properties(self):
        properties = [
            "staleness_manager",
            "workflow_executor",
            "dispatcher",
            "runner",
            "proxy_gateway_addr",
            "worker_ids",
        ]
        for p in properties:
            assert hasattr(GatewayRolloutController, p), f"Missing property: {p}"

    def test_not_subclass_of_rollout_controller(self):
        """GatewayRolloutController must NOT be a subclass of RolloutController."""
        # Verify it doesn't inherit from any class except object
        bases = GatewayRolloutController.__bases__
        assert bases == (object,), f"Unexpected bases: {bases}"


# =============================================================================
# GatewayRolloutController — construction + state
# =============================================================================


class TestGatewayRolloutControllerConstruction:
    def test_constructor(self):
        cfg = GatewayControllerConfig()
        scheduler = MagicMock()
        controller = GatewayRolloutController(config=cfg, scheduler=scheduler)

        assert controller.config is cfg
        assert controller.scheduler is scheduler
        assert controller.workers == []
        assert controller.server_infos == []
        assert controller.get_version() == 0
        assert controller.staleness_manager is None
        assert controller._worker_ids == {}
        assert controller.worker_ids == {}

    def test_version_management_without_services(self):
        """set_version / get_version work even without gateway services."""
        cfg = GatewayControllerConfig()
        scheduler = MagicMock()
        controller = GatewayRolloutController(config=cfg, scheduler=scheduler)

        # No gateway services started, but version management is local
        controller._version = 42
        assert controller.get_version() == 42

    def test_export_stats_returns_dict(self):
        cfg = GatewayControllerConfig()
        scheduler = MagicMock()
        controller = GatewayRolloutController(config=cfg, scheduler=scheduler)
        stats = controller.export_stats()
        assert isinstance(stats, dict)

    def test_start_proxy_is_noop(self):
        cfg = GatewayControllerConfig()
        scheduler = MagicMock()
        controller = GatewayRolloutController(config=cfg, scheduler=scheduler)
        # Should not raise
        controller.start_proxy()
        controller.start_proxy_gateway()

    def test_proxy_gateway_addr(self):
        cfg = GatewayControllerConfig()
        scheduler = MagicMock()
        controller = GatewayRolloutController(config=cfg, scheduler=scheduler)
        # Before initialize, proxy_gateway_addr returns the empty _gateway_addr
        assert controller.proxy_gateway_addr == ""

    def test_workflow_executor_raises_before_init(self):
        cfg = GatewayControllerConfig()
        scheduler = MagicMock()
        controller = GatewayRolloutController(config=cfg, scheduler=scheduler)
        with pytest.raises(RuntimeError, match="initialize"):
            _ = controller.workflow_executor

    def test_config_perf_tracer_is_noop(self):
        cfg = GatewayControllerConfig()
        scheduler = MagicMock()
        controller = GatewayRolloutController(config=cfg, scheduler=scheduler)
        # Should not raise
        controller.config_perf_tracer()
        controller.save_perf_tracer()


# =============================================================================
# GatewayRolloutController — gateway HTTP helpers
# =============================================================================


class TestGatewayRolloutControllerHTTP:
    def test_gateway_http_post_raises_on_failure(self):
        cfg = GatewayControllerConfig()
        scheduler = MagicMock()
        controller = GatewayRolloutController(config=cfg, scheduler=scheduler)
        # _gateway_addr points to unreachable host — should raise RuntimeError
        controller._gateway_addr = "http://127.0.0.1:19999"
        with pytest.raises(RuntimeError, match="Failed to POST"):
            controller._gateway_http_post("/test", {"key": "value"})

    @patch("requests.post")
    def test_gateway_http_post_sends_auth(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_post.return_value = mock_resp

        cfg = GatewayControllerConfig(admin_api_key="my-secret-key")
        scheduler = MagicMock()
        controller = GatewayRolloutController(config=cfg, scheduler=scheduler)
        controller._gateway_addr = "http://127.0.0.1:8080"

        controller._gateway_http_post("/test_endpoint", {"data": 1})

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert "Bearer my-secret-key" in str(call_kwargs)
        assert "http://127.0.0.1:8080/test_endpoint" in str(call_kwargs)
