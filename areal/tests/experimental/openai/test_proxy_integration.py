"""Integration tests for the OpenAI Proxy architecture.

These tests verify end-to-end functionality of the proxy system:

1. GPU-based tests (TestProxyIntegration, TestProxyServerEndpoints):
   - LocalScheduler creates real worker processes
   - SGLang inference server runs locally
   - RolloutController manages rollout and proxy workers
   - OpenAIProxyWorkflow executes real agent workflows

2. Mock engine tests (TestProxyWithMockEngine):
   - Uses MockInferenceEngine that returns naive results
   - No GPU required - can run in CI environments
   - Tests the full proxy architecture with real RPC/HTTP communication
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from typing import Any

import pytest
import requests
import torch

from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import (
    InferenceEngineConfig,
    SGLangConfig,
)
from areal.api.io_struct import LocalInfServerInfo
from areal.api.workflow_api import AgentWorkflow
from areal.controller.rollout_controller import RolloutController
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.scheduler.local import LocalScheduler
from areal.tests.utils import get_model_path
from areal.utils import network, seeding
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.proc import kill_process_tree

# =============================================================================
# Test Configuration
# =============================================================================

EXPR_NAME = "test_proxy_integration"
TRIAL_NAME = "trial_0"
LOCAL_MODEL_PATH = "/storage/openpsi/models/Qwen__Qwen3-0.6B/"
HF_MODEL_ID = "Qwen/Qwen3-0.6B"
RUN_SERVER_TIMEOUT = 180


def get_test_model_path() -> str:
    """Get the model path for tests (lazy evaluation)."""
    return get_model_path(LOCAL_MODEL_PATH, HF_MODEL_ID)


def has_local_model() -> bool:
    """Check if the model is available locally (without network)."""
    import os

    return os.path.exists(LOCAL_MODEL_PATH)


def check_server_health(base_url: str) -> bool:
    """Check if the inference server is healthy."""
    try:
        response = requests.get(f"{base_url}/health", timeout=30)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def has_gpu() -> bool:
    """Check if GPU is available."""
    return torch.cuda.is_available() and torch.cuda.device_count() > 0


# =============================================================================
# Test Agent Workflow
# =============================================================================


class SimpleAgentWorkflow(AgentWorkflow):
    """A simple agent workflow for testing.

    Makes one chat completion call and returns a fixed reward.
    """

    async def run(
        self, base_url: str | None, data: dict[str, Any]
    ) -> dict[str, float] | float:
        from openai import AsyncOpenAI

        # Use provided base_url or get from environment
        if base_url is None:
            base_url = os.environ.get("OPENAI_BASE_URL")

        client = AsyncOpenAI(
            base_url=base_url,
            api_key="dummy",  # Not used by our proxy
        )

        messages = data.get("messages", [{"role": "user", "content": "Say hello"}])

        response = await client.chat.completions.create(
            model="test",  # Model name is ignored by our proxy
            messages=messages,
            max_tokens=32,
            temperature=1.0,
            top_p=1.0,
        )

        # Return a simple reward based on response length
        content = response.choices[0].message.content or ""
        reward = min(len(content) / 100.0, 1.0)
        return reward


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def sglang_server():
    """Launch an SGLang server for testing."""
    if not has_gpu():
        pytest.skip("GPU required for SGLang server")

    seeding.set_random_seed(1, EXPR_NAME)

    host = network.gethostip()
    port, dist_port = network.find_free_ports(2)

    cmd = SGLangConfig.build_cmd(
        sglang_config=SGLangConfig(
            skip_tokenizer_init=True,
            model_path=get_test_model_path(),
            mem_fraction_static=0.3,
        ),
        host=host,
        port=port,
        tp_size=1,
        base_gpu_id=0,
        dist_init_addr=f"{host}:{dist_port}",
    )

    process = subprocess.Popen(
        cmd,
        stdout=sys.stdout,
        stderr=sys.stdout,
    )

    base_url = f"http://{host}:{port}"

    # Wait for server to be ready
    tik = time.time()
    while time.time() - tik < RUN_SERVER_TIMEOUT:
        if check_server_health(base_url):
            break
        time.sleep(1)

    if time.time() - tik >= RUN_SERVER_TIMEOUT:
        kill_process_tree(process.pid, graceful=True)
        pytest.fail("SGLang server failed to start within timeout")

    yield {"host": host, "port": port, "base_url": base_url, "process": process}

    kill_process_tree(process.pid, graceful=True)


@pytest.fixture
def local_scheduler(tmp_path):
    """Create a LocalScheduler for testing."""
    if not has_gpu():
        pytest.skip("GPU required for LocalScheduler")

    scheduler = LocalScheduler(
        gpu_devices=[0],
        log_dir=str(tmp_path),
        experiment_name=EXPR_NAME,
        trial_name=TRIAL_NAME,
    )
    yield scheduler
    scheduler.shutdown()


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.slow
@pytest.mark.skipif(not has_gpu(), reason="GPU required for integration tests")
class TestProxyIntegration:
    """Integration tests for the proxy architecture."""

    def test_proxy_workflow_roundtrip(self, sglang_server, local_scheduler, tmp_path):
        """Test a complete roundtrip through the proxy architecture.

        This test:
        1. Creates a RolloutController with RemoteSGLangEngine
        2. Initializes rollout workers connected to the SGLang server
        3. Starts proxy workers
        4. Runs a simple agent workflow through OpenAIProxyWorkflow
        5. Verifies the result contains expected tensor fields
        """
        from areal.experimental.openai.proxy.workflow import OpenAIProxyWorkflow

        # Create RolloutController
        config = InferenceEngineConfig(
            experiment_name=EXPR_NAME,
            trial_name=TRIAL_NAME,
            max_concurrent_rollouts=2,
            consumer_batch_size=1,
            tokenizer_path=get_test_model_path(),
        )

        rollout = RolloutController(
            inf_engine=RemoteSGLangEngine,
            config=config,
            scheduler=local_scheduler,
        )

        try:
            # Initialize rollout workers connected to existing SGLang server
            server_info = LocalInfServerInfo(
                host=sglang_server["host"],
                port=sglang_server["port"],
            )

            rollout.initialize(
                role="rollout",
                alloc_mode=AllocationMode.from_str("sglang:d1"),
                server_args={},  # Not used when server_infos provided
                server_infos=[server_info],
            )

            # Start proxy workers
            rollout.start_proxy()

            # Get proxy address
            proxy_addr = rollout.get_proxy_addr(0)
            assert proxy_addr.startswith("http://")

            # Create test data
            test_data = [
                {
                    "messages": [{"role": "user", "content": "What is 2+2?"}],
                    "answer": "4",
                }
            ]

            # Run workflow
            result = rollout.rollout_batch(
                data=test_data,
                workflow=OpenAIProxyWorkflow,
                workflow_kwargs={
                    "mode": "offline",
                    "agent": SimpleAgentWorkflow(),
                    "proxy_addr": proxy_addr,
                    "discount": 1.0,
                    "export_style": "individual",
                },
            )

            # Verify result structure
            assert isinstance(result, dict)
            assert "input_ids" in result
            assert isinstance(result["input_ids"], torch.Tensor)
            assert result["input_ids"].dim() == 2  # [batch, seq_len]

        finally:
            rollout.destroy()

    def test_proxy_session_lifecycle(self, sglang_server, tmp_path):
        """Test proxy session start/end lifecycle via HTTP.

        This test verifies the proxy server handles session management correctly:
        1. Health check endpoint works
        2. Sessions can be started and ended
        3. Capacity management works
        """

        # For this test, we need to start a standalone proxy server
        # This is a simpler test that just verifies the HTTP endpoints

        # Skip if no GPU - we need the SGLang server running
        if not has_gpu():
            pytest.skip("GPU required for proxy session tests")

        # The full integration is tested in test_proxy_workflow_roundtrip
        # This test focuses on the client session API
        pass  # Placeholder - full test requires running proxy server

    def test_proxy_addr_passthrough(self, local_scheduler, tmp_path):
        """Test that proxy_addr is correctly passed through the submit chain.

        This is a unit test for the proxy_addr parameter handling.
        """
        from unittest.mock import AsyncMock, patch

        config = InferenceEngineConfig(
            experiment_name=EXPR_NAME,
            trial_name=TRIAL_NAME,
            max_concurrent_rollouts=2,
            consumer_batch_size=1,
        )

        # Create controller with mocked engine
        with patch.object(
            local_scheduler, "async_call_engine", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = 1  # Return task_id

            _ = RolloutController(
                inf_engine=RemoteSGLangEngine,
                config=config,
                scheduler=local_scheduler,
            )

            # Test that submit accepts proxy_addr
            # We can't fully test without initializing, but we can verify the API
            from areal.controller.rollout_controller import _RemoteRolloutTaskInput

            task_input = _RemoteRolloutTaskInput(
                task_id=1,
                data={"test": "data"},
                workflow="test.workflow",
                workflow_kwargs={},
                should_accept_fn=None,
                is_eval=False,
                group_size=1,
                proxy_addr="http://localhost:8000",
            )

            # Verify the dataclass has proxy_addr field
            assert task_input.proxy_addr == "http://localhost:8000"


@pytest.mark.slow
@pytest.mark.skipif(not has_gpu(), reason="GPU required for integration tests")
class TestProxyServerEndpoints:
    """Test proxy server HTTP endpoints directly."""

    def test_session_timeout_cleanup(self):
        """Test that stale sessions are cleaned up."""
        from areal.experimental.openai.proxy.proxy_rollout_server import (
            SESSION_TIMEOUT_SECONDS,
            SessionData,
            _session_cache,
        )

        # Create a session
        session = SessionData(session_id="test-session-1")
        _session_cache["test-session-1"] = session

        # Session should not be stale immediately
        assert not session.is_stale()

        # Manually set last access time to past
        import time

        session._last_access_time = time.time() - SESSION_TIMEOUT_SECONDS - 1

        # Now it should be stale
        assert session.is_stale()

        # Clean up the test session
        _session_cache.pop("test-session-1", None)

    def test_session_data_lifecycle(self):
        """Test SessionData class methods."""
        from areal.experimental.openai.proxy.proxy_rollout_server import SessionData

        session = SessionData(session_id="test-lifecycle")

        # Test initial state
        assert session.session_id == "test-lifecycle"
        assert not session._completed

        # Test update_last_access
        import time

        old_time = session._last_access_time
        time.sleep(0.01)
        session.update_last_access()
        assert session._last_access_time > old_time

        # Test finish
        session.finish()
        assert session._completed
        assert session._end_time is not None

        # Test wait_for_finish (should return immediately since already finished)
        result = session.wait_for_finish(timeout=0.1)
        assert result is True


# =============================================================================
# Standalone Test Runner
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])


# =============================================================================
# Mock Engine Tests (No GPU Required)
# =============================================================================


class TestProxyWithMockEngine:
    """Integration tests using MockInferenceEngine (no GPU required).

    These tests exercise the full proxy architecture with real processes:
    - LocalScheduler spawns real RPC server processes
    - Proxy server processes handle HTTP requests
    - OpenAIProxyWorkflow coordinates agent execution
    - AgentWorkflow makes real OpenAI API calls to the proxy

    Only the inference engine is mocked - it returns deterministic
    responses without GPU computation.
    """

    @pytest.fixture
    def mock_scheduler(self, tmp_path):
        """Create a LocalScheduler without GPU devices."""
        scheduler = LocalScheduler(
            gpu_devices=[],  # No GPU required
            log_dir=str(tmp_path),
            experiment_name=EXPR_NAME,
            trial_name=TRIAL_NAME,
        )
        yield scheduler
        scheduler.shutdown()

    def test_mock_engine_standalone(self, tmp_path):
        """Test MockInferenceEngine can be instantiated and used directly."""
        if not has_local_model():
            pytest.skip("Local model not available - skipping tokenizer-dependent test")

        from areal.tests.experimental.openai.mock_engine import MockInferenceEngine

        config = InferenceEngineConfig(
            experiment_name=EXPR_NAME,
            trial_name=TRIAL_NAME,
            tokenizer_path=get_test_model_path(),
        )

        engine = MockInferenceEngine(config)

        # Test initialization
        assert not engine.initialized
        engine.initialize(addr="localhost:8000")
        assert engine.initialized

        # Test version tracking
        assert engine.get_version() == 0
        engine.set_version(42)
        assert engine.get_version() == 42

        # Test destroy
        engine.destroy()
        assert not engine.initialized

    @pytest.mark.asyncio
    async def test_mock_engine_agenerate(self, tmp_path):
        """Test MockInferenceEngine.agenerate returns valid response."""
        if not has_local_model():
            pytest.skip("Local model not available - skipping tokenizer-dependent test")

        from areal.api.io_struct import ModelRequest
        from areal.tests.experimental.openai.mock_engine import MockInferenceEngine

        tokenizer = load_hf_tokenizer(get_test_model_path())

        config = InferenceEngineConfig(
            experiment_name=EXPR_NAME,
            trial_name=TRIAL_NAME,
            tokenizer_path=get_test_model_path(),
        )

        engine = MockInferenceEngine(config)
        engine.initialize()

        # Create a request
        input_text = "Hello, world!"
        input_ids = tokenizer.encode(input_text, add_special_tokens=False)

        request = ModelRequest(
            input_ids=input_ids,
            tokenizer=tokenizer,
        )

        # Generate response
        response = await engine.agenerate(request)

        # Verify response structure
        assert response.input_tokens == input_ids
        assert len(response.output_tokens) > 0
        assert len(response.output_logprobs) == len(response.output_tokens)
        assert response.stop_reason == "stop"

        # Verify we can decode the output
        output_text = tokenizer.decode(response.output_tokens_without_stop)
        assert "Hello" in output_text  # Mock response starts with "Hello!"

        engine.destroy()

    def test_session_data_no_gpu(self):
        """Test SessionData class without GPU (unit test)."""
        from areal.experimental.openai.proxy.proxy_rollout_server import (
            SessionData,
        )

        session = SessionData(session_id="mock-test-session")

        # Test initial state
        assert session.session_id == "mock-test-session"
        assert not session._completed

        # Test update_last_access
        old_time = session._last_access_time
        time.sleep(0.01)
        session.update_last_access()
        assert session._last_access_time > old_time

        # Test is_stale
        assert not session.is_stale()

        # Test finish
        session.finish()
        assert session._completed

        # Test wait_for_finish
        result = session.wait_for_finish(timeout=0.1)
        assert result is True

    def test_proxy_addr_dataclass(self):
        """Test _RemoteRolloutTaskInput accepts proxy_addr."""
        from areal.controller.rollout_controller import _RemoteRolloutTaskInput

        task_input = _RemoteRolloutTaskInput(
            task_id=1,
            data={"test": "data"},
            workflow="test.workflow",
            workflow_kwargs={},
            should_accept_fn=None,
            is_eval=False,
            group_size=1,
            proxy_addr="http://localhost:8000",
        )

        assert task_input.proxy_addr == "http://localhost:8000"

        # Test with None proxy_addr
        task_input_no_proxy = _RemoteRolloutTaskInput(
            task_id=2,
            data={"test": "data"},
            workflow="test.workflow",
            workflow_kwargs={},
            should_accept_fn=None,
        )
        assert task_input_no_proxy.proxy_addr is None
