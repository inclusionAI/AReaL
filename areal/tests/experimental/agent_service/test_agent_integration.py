"""Integration tests for Agent Service with full five-process architecture.

This module tests the complete flow:
Controller -> Rollout Worker -> Agent Service -> Proxy Server -> SGLang
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest
import requests
import torch

from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import AgentServiceSpec, InferenceEngineConfig, SGLangConfig
from areal.api.io_struct import LocalInfServerInfo
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.experimental.agent_service.agent_controller import AgentController
from areal.experimental.agent_service.config import GatewayConfig
from areal.infra.controller.rollout_controller import RolloutController
from areal.infra.rpc.rtensor import RTensor
from areal.infra.scheduler.local import LocalScheduler
from areal.infra.utils.proc import kill_process_tree
from areal.tests.utils import get_model_path
from areal.utils import network, seeding

# =============================================================================
# Test Configuration
# =============================================================================

EXPR_NAME = "test_agent_integration"
TRIAL_NAME = "trial_0"
LOCAL_MODEL_PATH = "/storage/openpsi/models/Qwen__Qwen3-0.6B/"
HF_MODEL_ID = "Qwen/Qwen3-0.6B"
RUN_SERVER_TIMEOUT = 180


def get_test_model_path() -> str:
    """Get the model path for tests (lazy evaluation)."""
    return get_model_path(LOCAL_MODEL_PATH, HF_MODEL_ID)


def has_local_model() -> bool:
    """Check if the model is available locally (without network)."""
    return os.path.exists(LOCAL_MODEL_PATH)


def check_server_health(base_url: str) -> bool:
    """Check if the inference server is healthy."""
    try:
        response = requests.get(f"{base_url}/health", timeout=30)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def wait_for_agent_service(agent_addr: str, timeout: float = 60) -> None:
    """Wait for Agent Service to be ready.

    Parameters
    ----------
    agent_addr : str
        The Agent Service HTTP address (e.g., "http://localhost:8300").
    timeout : float
        Maximum time to wait in seconds.

    Raises
    ------
    TimeoutError
        If the Agent Service is not ready within the timeout.
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        if check_server_health(agent_addr):
            return
        time.sleep(0.5)
    raise TimeoutError(f"Agent Service at {agent_addr} not ready within {timeout}s")


def has_gpu() -> bool:
    """Check if GPU is available."""
    return torch.cuda.is_available() and torch.cuda.device_count() > 0


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

    fileroot = tmp_path / "fileroot"
    fileroot.mkdir()
    name_resolve_root = tmp_path / "name_resolve"
    name_resolve_root.mkdir()
    scheduler = LocalScheduler(
        gpu_devices=[0],
        log_dir=str(tmp_path),
        experiment_name=EXPR_NAME,
        trial_name=TRIAL_NAME,
        fileroot=str(fileroot),
        nfs_record_root=str(name_resolve_root),
    )
    yield scheduler
    scheduler.delete_workers(None)


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.slow
@pytest.mark.skipif(not has_gpu(), reason="GPU required for integration tests")
class TestAgentServiceIntegration:
    """Integration tests for Agent Service in the five-process architecture."""

    def test_agent_service_workflow_roundtrip(
        self, sglang_server, local_scheduler, tmp_path
    ):
        """Test a complete roundtrip through Agent Service.

        This test:
        1. Creates a RolloutController with RemoteSGLangEngine
        2. Initializes rollout workers connected to the SGLang server
        3. Starts proxy workers
        4. Starts agent service workers
        5. Runs a workflow through OpenAIProxyWorkflow with mode="service"
        6. Verifies the result contains expected tensor fields
        """
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

        agent_controller = AgentController(
            config=GatewayConfig(), scheduler=local_scheduler
        )

        try:
            # Initialize rollout workers connected to existing SGLang server
            server_info = LocalInfServerInfo(
                process=None,
                host=sglang_server["host"],
                port=sglang_server["port"],
            )

            rollout.initialize(
                role="rollout",
                alloc_mode=AllocationMode.from_str("sglang:d1"),
                server_infos=[server_info],
            )

            # Start proxy workers
            rollout.start_proxy()

            # Start agent service via AgentController (Trainer-level responsibility)
            spec = AgentServiceSpec(
                agent_import_path="areal.tests.experimental.openai.utils.SimpleAgent",
                agent_reuse=False,
                agent_init_kwargs={},
                workers=1,
            )
            agent_addr = agent_controller.start(spec)
            rollout.set_agent_service_addr(agent_addr)

            # Get proxy address
            proxy_addr = rollout.get_proxy_addr(0)
            assert proxy_addr.startswith("http://")
            assert agent_addr.startswith("http://")

            # Wait for Agent Service to be ready
            wait_for_agent_service(agent_addr)

            # Create test data
            test_data = [
                {
                    "messages": [{"role": "user", "content": "What is 2+2?"}],
                    "answer": "4",
                }
            ]

            # Run workflow with mode="service"
            result = rollout.rollout_batch(
                data=test_data,
                workflow="areal.experimental.openai.proxy.workflow.OpenAIProxyWorkflow",
                workflow_kwargs={
                    "mode": "service",
                    "proxy_addr": proxy_addr,
                },
            )

            # Verify result structure
            assert isinstance(result, dict)
            assert "input_ids" in result
            assert isinstance(result["input_ids"], RTensor)
            assert result["input_ids"].ndim == 2  # [batch, seq_len]

        finally:
            agent_controller.stop()
            rollout.destroy()

    def test_agent_service_shared_mode(self, sglang_server, local_scheduler, tmp_path):
        """Test Agent Service in shared mode (agent_reuse=True).

        This test verifies that:
        1. Agent Service can start in shared mode
        2. Workflow completes successfully with mode="service"

        Note: Multi-iteration testing is skipped due to a known issue with
        rollout_batch handling multiple sequential calls.
        """
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

        agent_controller = AgentController(
            config=GatewayConfig(), scheduler=local_scheduler
        )

        try:
            server_info = LocalInfServerInfo(
                process=None,
                host=sglang_server["host"],
                port=sglang_server["port"],
            )

            rollout.initialize(
                role="rollout",
                alloc_mode=AllocationMode.from_str("sglang:d1"),
                server_infos=[server_info],
            )

            rollout.start_proxy()

            proxy_addr = rollout.get_proxy_addr(0)

            # Start agent service in shared mode
            spec = AgentServiceSpec(
                agent_import_path="areal.tests.experimental.openai.utils.SimpleAgent",
                agent_reuse=True,
                agent_init_kwargs={"shared": True},
                workers=1,
            )
            agent_addr = agent_controller.start(spec)
            rollout.set_agent_service_addr(agent_addr)

            # Wait for Agent Service to be ready
            wait_for_agent_service(agent_addr)

            # Test data
            test_data = [
                {
                    "messages": [{"role": "user", "content": "Hello"}],
                    "answer": "Hi",
                }
            ]

            # Run single batch to verify shared mode works
            result = rollout.rollout_batch(
                data=test_data,
                workflow="areal.experimental.openai.proxy.workflow.OpenAIProxyWorkflow",
                workflow_kwargs={
                    "mode": "service",
                    "proxy_addr": proxy_addr,
                },
            )
            assert isinstance(result, dict)
            assert "input_ids" in result

        finally:
            agent_controller.stop()
            rollout.destroy()
