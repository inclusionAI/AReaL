"""Tests for GatewayInferenceController."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from areal.api.cli_args import OpenAIProxyConfig
from areal.experimental.inference_service.controller.config import (
    GatewayControllerConfig,
)
from areal.experimental.inference_service.controller.controller import (
    GatewayInferenceController,
)
from areal.experimental.inference_service.controller.workflow import (
    InferenceServiceWorkflow,
)

# =============================================================================
# GatewayControllerConfig
# =============================================================================


class TestGatewayControllerConfig:
    def test_defaults(self):
        cfg = GatewayControllerConfig()
        assert isinstance(cfg.openai, OpenAIProxyConfig)
        assert cfg.openai.admin_api_key == "areal-admin-key"
        assert cfg.consumer_batch_size == 16
        assert cfg.max_concurrent_rollouts is None
        assert cfg.max_head_offpolicyness == 0
        assert cfg.enable_rollout_tracing is False
        assert cfg.set_reward_finish_timeout == 0.0

    def test_custom_values(self):
        cfg = GatewayControllerConfig(
            openai=OpenAIProxyConfig(admin_api_key="custom-key"),
            consumer_batch_size=32,
            max_concurrent_rollouts=64,
            max_head_offpolicyness=5,
            set_reward_finish_timeout=3.0,
        )
        assert cfg.openai is not None
        assert cfg.openai.admin_api_key == "custom-key"
        assert cfg.consumer_batch_size == 32
        assert cfg.max_concurrent_rollouts == 64
        assert cfg.max_head_offpolicyness == 5
        assert cfg.set_reward_finish_timeout == 3.0

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
# GatewayInferenceController — workflow resolution helpers
# =============================================================================


class TestControllerWorkflowResolution:
    def test_resolve_workflow_with_instance(self):
        controller = GatewayInferenceController(
            config=GatewayControllerConfig(),
            scheduler=MagicMock(),
        )
        with pytest.raises(TypeError, match=r"callable run\(\) method"):
            controller._resolve_workflow(12345)

    def test_resolve_workflow_none_creates_online_inference_service_workflow(self):
        cfg = GatewayControllerConfig(
            openai=OpenAIProxyConfig(admin_api_key="test-admin-key")
        )
        scheduler = MagicMock()
        controller = GatewayInferenceController(config=cfg, scheduler=scheduler)
        controller._gateway_addr = "http://test:8080"

        resolved = controller._resolve_workflow(
            None,
            workflow_kwargs={"timeout": 3.0},
        )

        assert isinstance(resolved, InferenceServiceWorkflow)
        assert resolved.controller is controller
        assert resolved.agent is None
        assert resolved.timeout == 3.0

    def test_resolve_workflow_agent_class_creates_offline_workflow(self):
        cfg = GatewayControllerConfig(
            openai=OpenAIProxyConfig(admin_api_key="test-admin-key")
        )
        scheduler = MagicMock()
        controller = GatewayInferenceController(config=cfg, scheduler=scheduler)
        controller._gateway_addr = "http://test:8080"

        class MockAgent:
            async def run(self, data, **kwargs):
                return 1.0

        resolved = controller._resolve_workflow(
            MockAgent,
            workflow_kwargs={},
        )

        assert isinstance(resolved, InferenceServiceWorkflow)
        assert resolved.agent is not None
        assert isinstance(resolved.agent, MockAgent)

    def test_resolve_should_accept_fn_none(self):
        assert GatewayInferenceController._resolve_should_accept_fn(None) is None

    def test_resolve_should_accept_fn_callable(self):
        fn = lambda x: True  # noqa: E731
        assert GatewayInferenceController._resolve_should_accept_fn(fn) is fn

    def test_resolve_workflow_with_agent_class(self):
        """Test _resolve_workflow wraps agent-like classes in InferenceServiceWorkflow."""
        cfg = GatewayControllerConfig()
        scheduler = MagicMock()
        controller = GatewayInferenceController(config=cfg, scheduler=scheduler)
        controller._gateway_addr = "http://test:8080"

        class MockAgent:
            async def run(self, data, **kwargs):
                return 1.0

        resolved = controller._resolve_workflow(
            MockAgent,
            workflow_kwargs={},
        )
        assert isinstance(resolved, InferenceServiceWorkflow)
        assert resolved.agent is not None
        assert hasattr(resolved, "arun_episode")

    def test_resolve_workflow_agent_class_without_gateway_raises(self):
        controller = GatewayInferenceController(
            config=GatewayControllerConfig(),
            scheduler=MagicMock(),
        )

        class MockAgent:
            async def run(self, data, **kwargs):
                return 1.0

        with pytest.raises(ValueError, match="Gateway address is unavailable"):
            controller._resolve_workflow(MockAgent, workflow_kwargs={})

    def test_resolve_workflow_rollout_workflow_instance_raises(self):
        controller = GatewayInferenceController(
            config=GatewayControllerConfig(),
            scheduler=MagicMock(),
        )
        controller._gateway_addr = "http://test:8080"

        workflow = InferenceServiceWorkflow(
            controller=controller,
            gateway_addr="http://test:8080",
        )

        with pytest.raises(
            TypeError,
            match="direct RolloutWorkflow instances are not supported",
        ):
            controller._resolve_workflow(workflow)

    def test_resolve_workflow_rollout_workflow_class_raises(self):
        controller = GatewayInferenceController(
            config=GatewayControllerConfig(),
            scheduler=MagicMock(),
        )
        controller._gateway_addr = "http://test:8080"

        with pytest.raises(
            TypeError,
            match="direct RolloutWorkflow classes are not supported",
        ):
            controller._resolve_workflow(
                "areal.experimental.inference_service.controller.workflow.InferenceServiceWorkflow"
            )


# =============================================================================
# GatewayInferenceController — API surface
# =============================================================================


class TestGatewayInferenceControllerAPISurface:
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
            assert hasattr(GatewayInferenceController, m), f"Missing method: {m}"

    def test_has_properties(self):
        properties = [
            "staleness_manager",
            "workflow_executor",
            "dispatcher",
            "runner",
            "proxy_gateway_addr",
            "inference_worker_addrs",
            "worker_ids",
        ]
        for p in properties:
            assert hasattr(GatewayInferenceController, p), f"Missing property: {p}"

    def test_not_subclass_of_rollout_controller(self):
        """GatewayInferenceController must NOT be a subclass of RolloutController."""
        # Verify it doesn't inherit from any class except object
        bases = GatewayInferenceController.__bases__
        assert bases == (object,), f"Unexpected bases: {bases}"


# =============================================================================
# GatewayInferenceController — construction + state
# =============================================================================


class TestGatewayInferenceControllerConstruction:
    def test_constructor(self):
        cfg = GatewayControllerConfig()
        scheduler = MagicMock()
        controller = GatewayInferenceController(config=cfg, scheduler=scheduler)

        assert controller.config is cfg
        assert controller.scheduler is scheduler
        assert controller.workers == []
        assert controller.server_infos == []
        assert controller.get_version() == 0
        assert controller.staleness_manager is None
        assert controller._worker_ids == {}
        assert controller.worker_ids == {}

    def test_admin_api_key_defaults_from_openai_proxy_config(self):
        cfg = GatewayControllerConfig()
        scheduler = MagicMock()
        controller = GatewayInferenceController(config=cfg, scheduler=scheduler)
        assert controller.config.openai.admin_api_key == "areal-admin-key"

    def test_version_management_without_services(self):
        """set_version / get_version work even without gateway services."""
        cfg = GatewayControllerConfig()
        scheduler = MagicMock()
        controller = GatewayInferenceController(config=cfg, scheduler=scheduler)

        # No gateway services started, but version management is local
        controller._version = 42
        assert controller.get_version() == 42

    def test_export_stats_returns_dict(self):
        cfg = GatewayControllerConfig()
        scheduler = MagicMock()
        controller = GatewayInferenceController(config=cfg, scheduler=scheduler)
        stats = controller.export_stats()
        assert isinstance(stats, dict)

    def test_start_proxy_is_noop(self):
        cfg = GatewayControllerConfig()
        scheduler = MagicMock()
        controller = GatewayInferenceController(config=cfg, scheduler=scheduler)
        # Should not raise
        controller.start_proxy()
        controller.start_proxy_gateway()

    def test_proxy_gateway_addr(self):
        cfg = GatewayControllerConfig()
        scheduler = MagicMock()
        controller = GatewayInferenceController(config=cfg, scheduler=scheduler)
        # Before initialize, proxy_gateway_addr returns the empty _gateway_addr
        assert controller.proxy_gateway_addr == ""

    def test_inference_worker_addrs_returns_copy(self):
        cfg = GatewayControllerConfig()
        scheduler = MagicMock()
        controller = GatewayInferenceController(config=cfg, scheduler=scheduler)
        controller._inf_addrs = ["http://127.0.0.1:30000"]

        result = controller.inference_worker_addrs
        assert result == ["http://127.0.0.1:30000"]
        result.append("http://127.0.0.1:30001")
        assert controller.inference_worker_addrs == ["http://127.0.0.1:30000"]


class TestGatewayInferenceControllerSGLangLaunch:
    @pytest.mark.asyncio
    async def test_async_initialize_uses_areal_sglang_server_module(self):
        from areal.api.cli_args import OpenAIProxyConfig, SchedulingSpec

        worker = MagicMock()
        worker.ip = "127.0.0.1"
        worker.worker_ports = [18000]
        worker.id = "w0"

        scheduler = MagicMock()
        scheduler.get_workers.return_value = [worker]

        cfg = GatewayControllerConfig(
            tokenizer_path="mock-tokenizer",
            model_path="mock-model",
            backend="sglang:d1",
            scheduling_spec=(
                SchedulingSpec(
                    gpu=0,
                    cpu=1,
                    mem=1,
                    cmd="python -m areal.experimental.inference_service.guard",
                ),
            ),
            openai=OpenAIProxyConfig(admin_api_key="test-admin-key"),
            setup_timeout=1.0,
        )
        controller = GatewayInferenceController(config=cfg, scheduler=scheduler)
        controller._callback_host = "127.0.0.1"
        controller._callback_port = 19000

        def _mock_resp(status_code=200, payload=None):
            r = MagicMock()
            r.status_code = status_code
            r.json.return_value = payload or {}
            r.raise_for_status.return_value = None
            return r

        with (
            patch.object(controller, "_wait_for_service") as mock_wait,
            patch.object(controller, "_fork_on_guard") as mock_fork,
            patch("requests.post") as mock_post,
            patch(
                "areal.api.cli_args.pkg_version.is_version_greater_or_equal",
                return_value=True,
            ),
            patch("areal.api.cli_args.is_version_less", return_value=False),
        ):
            mock_wait.return_value = None
            mock_fork.side_effect = [
                ("127.0.0.1", 18081),
                ("127.0.0.1", 18082),
                ("127.0.0.1", 18080),
            ]

            mock_post.side_effect = [
                _mock_resp(payload={"host": "127.0.0.1", "ports": [30000]}),
                _mock_resp(payload={"status": "success"}),
            ]

            await controller._async_initialize(server_args=None, server_infos=None)

        fork_payload = mock_post.call_args_list[1].kwargs["json"]
        raw_cmd = fork_payload["raw_cmd"]
        assert "-m" in raw_cmd
        assert "areal.engine.sglang_ext.areal_sglang_server" in raw_cmd

    def test_callback_addr_formats_ipv6_hostport(self):
        cfg = GatewayControllerConfig()
        scheduler = MagicMock()
        controller = GatewayInferenceController(config=cfg, scheduler=scheduler)
        controller._callback_host = "2001:db8::10"
        controller._callback_port = 19000

        assert controller.callback_addr == "[2001:db8::10]:19000"

    def test_workflow_executor_raises_before_init(self):
        cfg = GatewayControllerConfig()
        scheduler = MagicMock()
        controller = GatewayInferenceController(config=cfg, scheduler=scheduler)
        with pytest.raises(RuntimeError, match="initialize"):
            _ = controller.workflow_executor

    def test_config_perf_tracer_is_noop(self):
        cfg = GatewayControllerConfig()
        scheduler = MagicMock()
        controller = GatewayInferenceController(config=cfg, scheduler=scheduler)
        # Should not raise
        controller.config_perf_tracer()
        controller.save_perf_tracer()

    @pytest.mark.asyncio
    async def test_async_initialize_passes_callback_and_reward_timeout_to_data_proxy(
        self,
    ):
        from areal.api.cli_args import SchedulingSpec
        from areal.api.io_struct import LocalInfServerInfo

        worker = MagicMock()
        worker.ip = "127.0.0.1"
        worker.worker_ports = [18000]

        scheduler = MagicMock()
        scheduler.get_workers.return_value = [worker]

        cfg = GatewayControllerConfig(
            tokenizer_path="mock-tokenizer",
            request_timeout=15.0,
            set_reward_finish_timeout=7.5,
            scheduling_spec=(
                SchedulingSpec(
                    gpu=0,
                    cpu=1,
                    mem=1,
                    cmd="python -m areal.experimental.inference_service.guard",
                ),
            ),
            openai=OpenAIProxyConfig(admin_api_key="test-admin-key"),
        )
        controller = GatewayInferenceController(config=cfg, scheduler=scheduler)
        controller._callback_host = "127.0.0.1"
        controller._callback_port = 19000

        with patch.object(controller, "_fork_on_guard") as mock_fork:
            mock_fork.side_effect = [
                ("127.0.0.1", 18081),
                ("127.0.0.1", 18082),
                ("127.0.0.1", 18080),
            ]

            await controller._async_initialize(
                server_args=None,
                server_infos=[
                    LocalInfServerInfo(
                        host="127.0.0.1", port=30000, process=MagicMock()
                    )
                ],
            )

        data_proxy_cmd = mock_fork.call_args_list[1].kwargs["raw_cmd"]
        assert "--set-reward-finish-timeout" in data_proxy_cmd
        assert "7.5" in data_proxy_cmd
        assert "--callback-server-addr" in data_proxy_cmd
        assert "http://127.0.0.1:19000" in data_proxy_cmd


# =============================================================================
# GatewayInferenceController — gateway HTTP helpers
# =============================================================================


class TestGatewayInferenceControllerHTTP:
    def test_gateway_http_post_raises_on_failure(self):
        cfg = GatewayControllerConfig()
        scheduler = MagicMock()
        controller = GatewayInferenceController(config=cfg, scheduler=scheduler)
        # _gateway_addr points to unreachable host — should raise RuntimeError
        controller._gateway_addr = "http://127.0.0.1:19999"
        with pytest.raises(RuntimeError, match="Failed to POST"):
            controller._gateway_http_post("/test", {"key": "value"})

    @patch("requests.post")
    def test_gateway_http_post_sends_auth(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_post.return_value = mock_resp

        cfg = GatewayControllerConfig(
            openai=OpenAIProxyConfig(admin_api_key="my-secret-key")
        )
        scheduler = MagicMock()
        controller = GatewayInferenceController(config=cfg, scheduler=scheduler)
        controller._gateway_addr = "http://127.0.0.1:8080"

        controller._gateway_http_post("/test_endpoint", {"data": 1})

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert "Bearer my-secret-key" in str(call_kwargs)
        assert "http://127.0.0.1:8080/test_endpoint" in str(call_kwargs)


class TestOnlineCallbackFlow:
    @pytest.mark.asyncio
    async def test_online_callback_without_waiter_buffers_export_request(self):
        cfg = GatewayControllerConfig(
            openai=OpenAIProxyConfig(admin_api_key="test-admin-key")
        )
        scheduler = MagicMock()
        controller = GatewayInferenceController(config=cfg, scheduler=scheduler)
        controller._start_online_callback_server()
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"http://{controller.callback_addr}/callback/online_ready",
                    json={"session_id": "agent-a", "trajectory_id": 0},
                    headers={"Authorization": "Bearer test-admin-key"},
                )
            assert resp.status_code == 200
            buffered = await controller.wait_for_online_trajectory(timeout=1.0)
            assert buffered == {"session_id": "agent-a", "trajectory_id": 0}
        finally:
            controller._stop_online_callback_server()

    @pytest.mark.asyncio
    async def test_online_callback_settles_waiter_once(self):
        cfg = GatewayControllerConfig(
            openai=OpenAIProxyConfig(admin_api_key="test-admin-key")
        )
        scheduler = MagicMock()
        controller = GatewayInferenceController(config=cfg, scheduler=scheduler)
        controller._start_online_callback_server()

        waiter_task = asyncio.create_task(
            controller.wait_for_online_trajectory(timeout=1.0)
        )
        await asyncio.sleep(0)

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"http://{controller.callback_addr}/callback/online_ready",
                    json={"session_id": "agent-a", "trajectory_id": 0},
                    headers={"Authorization": "Bearer test-admin-key"},
                )
            assert resp.status_code == 200
            result = await waiter_task
            assert result == {"session_id": "agent-a", "trajectory_id": 0}
        finally:
            controller._stop_online_callback_server()

    @pytest.mark.asyncio
    async def test_online_callback_invalid_payload_keeps_waiter_pending(self):
        cfg = GatewayControllerConfig(
            openai=OpenAIProxyConfig(admin_api_key="test-admin-key")
        )
        scheduler = MagicMock()
        controller = GatewayInferenceController(config=cfg, scheduler=scheduler)
        controller._start_online_callback_server()

        waiter_task = asyncio.create_task(
            controller.wait_for_online_trajectory(timeout=1.0)
        )
        await asyncio.sleep(0)

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"http://{controller.callback_addr}/callback/online_ready",
                    json={"session_id": "agent-a"},
                    headers={"Authorization": "Bearer test-admin-key"},
                )
            assert resp.status_code == 425
            assert not waiter_task.done()
            waiter_task.cancel()
        finally:
            controller._stop_online_callback_server()

    @pytest.mark.asyncio
    async def test_cancelled_waiter_buffers_completed_online_result(self):
        cfg = GatewayControllerConfig(
            openai=OpenAIProxyConfig(admin_api_key="test-admin-key")
        )
        scheduler = MagicMock()
        controller = GatewayInferenceController(config=cfg, scheduler=scheduler)
        controller._start_online_callback_server()

        waiter_task = asyncio.create_task(
            controller.wait_for_online_trajectory(timeout=1.0)
        )
        await asyncio.sleep(0)
        waiter_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter_task

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"http://{controller.callback_addr}/callback/online_ready",
                    json={"session_id": "agent-a", "trajectory_id": 0},
                    headers={"Authorization": "Bearer test-admin-key"},
                )
            assert resp.status_code == 200

            buffered = await controller.wait_for_online_trajectory(timeout=1.0)
            assert buffered == {"session_id": "agent-a", "trajectory_id": 0}
        finally:
            controller._stop_online_callback_server()


class TestInferenceServiceWorkflow:
    @pytest.mark.asyncio
    async def test_online_mode_waits_on_controller(self):
        mock_interaction = MagicMock(reward=1.0)
        controller = MagicMock()
        controller.wait_for_online_trajectory = AsyncMock(
            return_value={"session_id": "sess-1", "trajectory_id": 7}
        )

        workflow = InferenceServiceWorkflow(
            controller=controller,
            agent=None,
            gateway_addr="http://test:8080",
            admin_api_key="test-key",
            timeout=3.0,
        )
        workflow._grant_capacity = AsyncMock()
        workflow._export_interactions = AsyncMock(
            return_value={"chatcmpl-1": mock_interaction}
        )

        with (
            patch(
                "areal.experimental.inference_service.controller.workflow.workflow_context"
            ) as mock_wf_ctx,
            patch(
                "areal.experimental.inference_service.controller.workflow.stats_tracker"
            ) as mock_st,
        ):
            mock_http_session = AsyncMock()
            mock_wf_ctx.get_aiohttp_session = AsyncMock(return_value=mock_http_session)
            mock_wf_ctx.stat_scope.return_value = "rollout"
            mock_st.get.return_value = MagicMock()

            result = await workflow.arun_episode(engine=MagicMock(), data={})

        assert result is not None
        assert "chatcmpl-1" in result
        workflow._grant_capacity.assert_awaited_once()
        controller.wait_for_online_trajectory.assert_awaited_once_with(timeout=3.0)
        workflow._export_interactions.assert_awaited_once_with(
            mock_http_session,
            "sess-1",
            trajectory_id=7,
        )

    @pytest.mark.asyncio
    async def test_offline_mode_runs_agent(self):
        controller = MagicMock()

        class MockAgent:
            async def run(self, data, **kwargs):
                return 1.0

        mock_interaction = MagicMock(reward=1.0)
        workflow = InferenceServiceWorkflow(
            controller=controller,
            agent=MockAgent(),
            gateway_addr="http://test:8080",
            admin_api_key="test-key",
        )
        workflow._grant_capacity = AsyncMock()
        workflow._start_session = AsyncMock(return_value=("sess-1", "sess-api-key-1"))
        workflow._set_last_reward = AsyncMock(return_value=None)
        workflow._export_interactions = AsyncMock(
            return_value={"chatcmpl-1": mock_interaction}
        )

        with (
            patch(
                "areal.experimental.inference_service.controller.workflow.workflow_context"
            ) as mock_wf_ctx,
            patch(
                "areal.experimental.inference_service.controller.workflow.stats_tracker"
            ) as mock_st,
        ):
            mock_http_session = AsyncMock()
            mock_wf_ctx.get_aiohttp_session = AsyncMock(return_value=mock_http_session)
            mock_wf_ctx.get.return_value = MagicMock(task_id=42)
            mock_wf_ctx.get_httpx_client = AsyncMock(return_value=MagicMock())
            mock_wf_ctx.stat_scope.return_value = "rollout"
            mock_st.get.return_value = MagicMock()

            result = await workflow.arun_episode(engine=MagicMock(), data={})

        assert result is not None
        assert "chatcmpl-1" in result
        workflow._grant_capacity.assert_awaited_once()
        workflow._start_session.assert_awaited_once()
        workflow._set_last_reward.assert_awaited_once_with(
            mock_http_session, 1.0, "sess-api-key-1"
        )
        workflow._export_interactions.assert_awaited_once_with(
            mock_http_session, "sess-1", trajectory_id=None
        )
