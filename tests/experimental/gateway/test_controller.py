"""Tests for GatewayRolloutController and GatewayInfEngine."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from areal.experimental.gateway.controller.config import GatewayControllerConfig
from areal.experimental.gateway.controller.inf_engine import (
    GatewayInfEngine,
    _GatewayInfEngineConfig,
)


# =============================================================================
# GatewayControllerConfig
# =============================================================================


class TestGatewayControllerConfig:
    def test_defaults(self):
        cfg = GatewayControllerConfig()
        assert cfg.gateway_host == "0.0.0.0"
        assert cfg.gateway_port == 8080
        assert cfg.router_port == 8081
        assert cfg.data_proxy_base_port == 8082
        assert cfg.admin_api_key == "areal-admin-key"
        assert cfg.consumer_batch_size == 16
        assert cfg.max_concurrent_rollouts is None
        assert cfg.max_head_offpolicyness == 0
        assert cfg.enable_rollout_tracing is False

    def test_custom_values(self):
        cfg = GatewayControllerConfig(
            gateway_port=9090,
            router_port=9091,
            admin_api_key="custom-key",
            consumer_batch_size=32,
            max_concurrent_rollouts=64,
            max_head_offpolicyness=5,
        )
        assert cfg.gateway_port == 9090
        assert cfg.router_port == 9091
        assert cfg.admin_api_key == "custom-key"
        assert cfg.consumer_batch_size == 32
        assert cfg.max_concurrent_rollouts == 64
        assert cfg.max_head_offpolicyness == 5

    def test_scheduling_fields(self):
        cfg = GatewayControllerConfig(
            request_timeout=60.0,
            setup_timeout=600.0,
            max_resubmit_retries=10,
        )
        assert cfg.request_timeout == 60.0
        assert cfg.setup_timeout == 600.0
        assert cfg.max_resubmit_retries == 10


# =============================================================================
# _GatewayInfEngineConfig adapter
# =============================================================================


class TestGatewayInfEngineConfigAdapter:
    def test_proxies_fields(self):
        cfg = GatewayControllerConfig(
            consumer_batch_size=32,
            max_concurrent_rollouts=64,
            enable_rollout_tracing=True,
        )
        adapter = _GatewayInfEngineConfig(cfg)
        assert adapter.consumer_batch_size == 32
        assert adapter.max_concurrent_rollouts == 64
        assert adapter.enable_rollout_tracing is True
        assert adapter.max_head_offpolicyness == 0
        assert adapter.queue_size is None


# =============================================================================
# GatewayInfEngine
# =============================================================================


class TestGatewayInfEngine:
    def test_construction(self):
        cfg = GatewayControllerConfig()
        engine = GatewayInfEngine("http://localhost:8080", cfg)
        assert engine.gateway_addr == "http://localhost:8080"
        assert engine.get_version() == 0
        assert engine.initialized is False

    def test_version_management(self):
        cfg = GatewayControllerConfig()
        engine = GatewayInfEngine("http://localhost:8080", cfg)
        engine.set_version(42)
        assert engine.get_version() == 42
        engine.set_version(0)
        assert engine.get_version() == 0

    def test_has_required_methods(self):
        """Verify GatewayInfEngine has all methods needed by WorkflowExecutor."""
        methods = [
            "agenerate",
            "submit",
            "wait",
            "wait_for_task",
            "rollout_batch",
            "prepare_batch",
            "set_version",
            "get_version",
            "pause",
            "resume",
            "initialize",
            "destroy",
        ]
        for m in methods:
            assert hasattr(GatewayInfEngine, m), f"Missing method: {m}"

    def test_workflow_executor_raises_before_init(self):
        cfg = GatewayControllerConfig()
        engine = GatewayInfEngine("http://localhost:8080", cfg)
        with pytest.raises(RuntimeError, match="not initialized"):
            _ = engine.workflow_executor

    def test_resolve_workflow_with_instance(self):
        """Test _resolve_workflow raises for non-RolloutWorkflow instances without proxy."""
        # Non-RolloutWorkflow instances hit case 5 (agent-like) and need proxy_addr + engine.
        with pytest.raises(ValueError, match="proxy_addr and engine are required"):
            GatewayInfEngine._resolve_workflow(12345)

    def test_resolve_workflow_none_raises(self):
        with pytest.raises(ValueError, match="must be specified"):
            GatewayInfEngine._resolve_workflow(None)

    def test_resolve_should_accept_fn_none(self):
        assert GatewayInfEngine._resolve_should_accept_fn(None) is None

    def test_resolve_should_accept_fn_callable(self):
        fn = lambda x: True  # noqa: E731
        assert GatewayInfEngine._resolve_should_accept_fn(fn) is fn

    def test_resolve_workflow_with_agent_class(self):
        """Test _resolve_workflow wraps agent-like classes in OpenAIProxyWorkflow."""
        cfg = GatewayControllerConfig()
        engine = GatewayInfEngine("http://test:8080", cfg)

        class MockAgent:
            async def run(self, data, **kwargs):
                return 1.0

        resolved = GatewayInfEngine._resolve_workflow(
            MockAgent,
            workflow_kwargs={},
            proxy_addr="http://test:8080",
            engine=engine,
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

        with pytest.raises(ValueError, match="proxy_addr and engine are required"):
            GatewayInfEngine._resolve_workflow(MockAgent, workflow_kwargs={})


# =============================================================================
# GatewayRolloutController — API surface
# =============================================================================


class TestGatewayRolloutControllerAPISurface:
    def test_has_all_public_methods(self):
        from areal.experimental.gateway.controller.controller import (
            GatewayRolloutController,
        )

        methods = [
            "initialize",
            "destroy",
            "submit",
            "wait",
            "rollout_batch",
            "prepare_batch",
            "agenerate",
            "set_version",
            "get_version",
            "get_capacity",
            "pause",
            "resume",
            "export_stats",
            "init_weights_update_group",
            "update_weights_from_distributed",
            "update_weights_from_disk",
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
        from areal.experimental.gateway.controller.controller import (
            GatewayRolloutController,
        )

        properties = [
            "callback_addr",
            "staleness_manager",
            "dispatcher",
            "runner",
            "proxy_gateway_addr",
        ]
        for p in properties:
            assert hasattr(GatewayRolloutController, p), f"Missing property: {p}"

    def test_not_subclass_of_rollout_controller(self):
        """GatewayRolloutController must NOT be a subclass of RolloutController."""
        from areal.experimental.gateway.controller.controller import (
            GatewayRolloutController,
        )

        # Verify it doesn't inherit from any class except object
        bases = GatewayRolloutController.__bases__
        assert bases == (object,), f"Unexpected bases: {bases}"


# =============================================================================
# GatewayRolloutController — construction + state
# =============================================================================


class TestGatewayRolloutControllerConstruction:
    def test_constructor(self):
        from areal.experimental.gateway.controller.controller import (
            GatewayRolloutController,
        )

        cfg = GatewayControllerConfig()
        scheduler = MagicMock()
        controller = GatewayRolloutController(config=cfg, scheduler=scheduler)

        assert controller.config is cfg
        assert controller.scheduler is scheduler
        assert controller.workers == []
        assert controller.server_infos == []
        assert controller.get_version() == 0
        assert controller.staleness_manager is None

    def test_version_management_without_services(self):
        """set_version / get_version work even without gateway services."""
        from areal.experimental.gateway.controller.controller import (
            GatewayRolloutController,
        )

        cfg = GatewayControllerConfig()
        scheduler = MagicMock()
        controller = GatewayRolloutController(config=cfg, scheduler=scheduler)

        # No gateway services started, but version management is local
        controller._version = 42
        assert controller.get_version() == 42

    def test_export_stats_returns_dict(self):
        from areal.experimental.gateway.controller.controller import (
            GatewayRolloutController,
        )

        cfg = GatewayControllerConfig()
        scheduler = MagicMock()
        controller = GatewayRolloutController(config=cfg, scheduler=scheduler)
        stats = controller.export_stats()
        assert isinstance(stats, dict)

    def test_start_proxy_is_noop(self):
        from areal.experimental.gateway.controller.controller import (
            GatewayRolloutController,
        )

        cfg = GatewayControllerConfig()
        scheduler = MagicMock()
        controller = GatewayRolloutController(config=cfg, scheduler=scheduler)
        # Should not raise
        controller.start_proxy()
        controller.start_proxy_gateway()

    def test_proxy_gateway_addr(self):
        from areal.experimental.gateway.controller.controller import (
            GatewayRolloutController,
        )

        cfg = GatewayControllerConfig(gateway_port=9999)
        scheduler = MagicMock()
        controller = GatewayRolloutController(config=cfg, scheduler=scheduler)
        assert controller.proxy_gateway_addr == "http://127.0.0.1:9999"

    def test_engine_raises_before_init(self):
        from areal.experimental.gateway.controller.controller import (
            GatewayRolloutController,
        )

        cfg = GatewayControllerConfig()
        scheduler = MagicMock()
        controller = GatewayRolloutController(config=cfg, scheduler=scheduler)
        with pytest.raises(RuntimeError, match="initialize"):
            _ = controller._engine

    def test_callback_addr_returns_gateway_address(self):
        """callback_addr should return gateway host:port without requiring a callback server."""
        from areal.experimental.gateway.controller.controller import (
            GatewayRolloutController,
        )

        cfg = GatewayControllerConfig()
        scheduler = MagicMock()
        controller = GatewayRolloutController(config=cfg, scheduler=scheduler)
        addr = controller.callback_addr
        assert ":" in addr
        host, port = addr.rsplit(":", 1)
        assert int(port) == cfg.gateway_port
        assert host != "0.0.0.0"

    def test_config_perf_tracer_is_noop(self):
        from areal.experimental.gateway.controller.controller import (
            GatewayRolloutController,
        )

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
    def test_gateway_http_post_logs_on_failure(self):
        from areal.experimental.gateway.controller.controller import (
            GatewayRolloutController,
        )

        cfg = GatewayControllerConfig(gateway_port=19999)
        scheduler = MagicMock()
        controller = GatewayRolloutController(config=cfg, scheduler=scheduler)

        # Should not raise — just logs the error
        controller._gateway_http_post("/test", {"key": "value"})

    @patch("requests.post")
    def test_gateway_http_post_sends_auth(self, mock_post):
        from areal.experimental.gateway.controller.controller import (
            GatewayRolloutController,
        )

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_post.return_value = mock_resp

        cfg = GatewayControllerConfig(gateway_port=8080, admin_api_key="my-secret-key")
        scheduler = MagicMock()
        controller = GatewayRolloutController(config=cfg, scheduler=scheduler)

        controller._gateway_http_post("/test_endpoint", {"data": 1})

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert "Bearer my-secret-key" in str(call_kwargs)
        assert "http://127.0.0.1:8080/test_endpoint" in str(call_kwargs)
