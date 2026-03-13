"""Integration tests for AEnvironment adapter with real service.

This module provides self-contained integration tests that automatically
start a local aenv MCP server for testing. No manual server startup required.

Environment variables:
    RUN_AENV_INTEGRATION: Set to "1" to enable these tests (required)
    AENV_URL: Base URL for aenv server (default: http://localhost)
    AENV_PORT: Port for aenv MCP server (default: 8081)
    AENV_EXAMPLES_DIR: Path to aenv examples directory (default: sibling AEnvironment/aenv/examples/all_in_one)
    AENV_TOOL_NAME: Tool name to test (optional, enables tool call test)
    AENV_TOOL_ARGS_JSON: JSON arguments for tool call (default: {})
"""

from __future__ import annotations

import atexit
import json
import os
import subprocess
import sys
import time
from collections.abc import Generator
from pathlib import Path

import pytest

from areal.infra.aenv import AenvConfig, AenvEnvironmentAdapter

RUN_AENV_INTEGRATION = os.getenv("RUN_AENV_INTEGRATION") == "1"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not RUN_AENV_INTEGRATION,
        reason="Set RUN_AENV_INTEGRATION=1 to run AEnvironment integration tests",
    ),
]


def _find_aenv_examples_dir() -> Path | None:
    """Find aenv examples directory.

    Search order:
    1. AENV_EXAMPLES_DIR environment variable
    2. Sibling directory ../AEnvironment/aenv/examples/all_in_one
    3. Common development paths
    """
    # 1. Check environment variable
    env_dir = os.getenv("AENV_EXAMPLES_DIR")
    if env_dir:
        path = Path(env_dir)
        if path.exists():
            return path

    # 2. Check sibling directory (common monorepo layout)
    current_file = Path(__file__).resolve()
    sibling_path = (
        current_file.parents[3] / "AEnvironment" / "aenv" / "examples" / "all_in_one"
    )
    if sibling_path.exists():
        return sibling_path

    # 3. Check if aenv is installed and find its examples
    try:
        import aenv

        aenv_path = Path(aenv.__file__).parent
        # aenv package is at aenv/src/aenv, examples are at aenv/examples
        examples_path = aenv_path.parents[1] / "examples" / "all_in_one"
        if examples_path.exists():
            return examples_path
    except ImportError:
        pass

    return None


def _wait_for_server(url: str, timeout: float = 30.0) -> bool:
    """Wait for server to be ready.

    Args:
        url: Base URL of the server
        timeout: Maximum time to wait in seconds

    Returns:
        True if server is ready, False otherwise
    """
    import urllib.error
    import urllib.request

    health_url = f"{url}/health"
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            req = urllib.request.Request(health_url, method="GET")
            with urllib.request.urlopen(req, timeout=2) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode())
                    if data.get("success") or data.get("status") in (
                        "success",
                        "healthy",
                    ):
                        return True
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            json.JSONDecodeError,
            TimeoutError,
        ):
            pass
        time.sleep(0.5)

    return False


@pytest.fixture(scope="module")
def _aenv_sdk_available():
    """Skip module if aenvironment SDK is not installed."""
    return pytest.importorskip("aenv")


@pytest.fixture(scope="module")
def aenv_server(_aenv_sdk_available) -> Generator[dict[str, str | int], None, None]:
    """Start a local aenv MCP server for testing.

    This fixture automatically:
    1. Finds the aenv examples directory
    2. Starts the MCP server on the specified port
    3. Waits for the server to be ready
    4. Provides server info to tests
    5. Cleans up the server process after tests

    Uses atexit to ensure cleanup even if pytest crashes before fixture teardown.

    Yields:
        Dict with server info: {"url": str, "port": int, "process": subprocess.Popen}
    """
    # Find examples directory
    examples_dir = _find_aenv_examples_dir()
    if examples_dir is None:
        pytest.skip(
            "Could not find aenv examples directory. Set AENV_EXAMPLES_DIR environment variable."
        )

    src_dir = examples_dir / "src"
    if not src_dir.exists():
        pytest.skip(f"aenv examples src directory not found: {src_dir}")

    # Get port from environment or use default
    port = int(os.getenv("AENV_PORT", "8081"))
    url = f"http://localhost:{port}"

    # Start server process
    process = subprocess.Popen(
        [sys.executable, "-m", "aenv.main", str(src_dir), "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Track cleanup state to avoid double-cleanup
    cleanup_done = False

    def _cleanup_process() -> None:
        """Terminate the server process if still running."""
        nonlocal cleanup_done
        if cleanup_done:
            return
        cleanup_done = True

        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    # Register cleanup with atexit to handle pytest crashes
    atexit.register(_cleanup_process)

    try:
        # Wait for server to be ready
        if not _wait_for_server(url, timeout=30.0):
            # Capture error output for debugging
            stdout, stderr = "", ""
            try:
                stdout, stderr = process.communicate(timeout=1)
            except subprocess.TimeoutExpired:
                pass
            # Cleanup before failing
            _cleanup_process()
            atexit.unregister(_cleanup_process)
            pytest.fail(
                f"AEnv MCP server failed to start on {url}.\n"
                f"stdout: {stdout}\n"
                f"stderr: {stderr}"
            )

        # Set DUMMY_INSTANCE_IP so tests connect directly to local server
        os.environ["DUMMY_INSTANCE_IP"] = "localhost"

        yield {"url": url, "port": port, "process": process}

    finally:
        # Normal cleanup via fixture teardown
        _cleanup_process()
        atexit.unregister(_cleanup_process)


@pytest.fixture(scope="module")
def aenv_config(aenv_server) -> AenvConfig:
    """Build integration config from server info and environment variables."""
    return AenvConfig(
        aenv_url=os.getenv("AENV_URL", "http://localhost"),
        env_name=os.getenv("AENV_ENV_NAME", "default"),
        datasource=os.getenv("AENV_DATASOURCE", ""),
        ttl=os.getenv("AENV_TTL", "30m"),
        timeout=float(os.getenv("AENV_TIMEOUT", "30")),
        startup_timeout=float(os.getenv("AENV_STARTUP_TIMEOUT", "120")),
        tool_call_timeout=float(os.getenv("AENV_TOOL_CALL_TIMEOUT", "30")),
        max_retries=int(os.getenv("AENV_MAX_RETRIES", "2")),
        retry_delay=float(os.getenv("AENV_RETRY_DELAY", "0.5")),
        auto_release=True,
    )


def _load_tool_args_from_env() -> dict:
    raw = os.getenv("AENV_TOOL_ARGS_JSON", "{}")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"AENV_TOOL_ARGS_JSON must be valid JSON: {raw}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("AENV_TOOL_ARGS_JSON must decode to a JSON object")
    return parsed


@pytest.mark.asyncio
async def test_aenv_environment_lifecycle_and_list_tools(aenv_config):
    """Validate initialize/list_tools/release with a live AEnvironment service."""
    async with AenvEnvironmentAdapter(aenv_config) as adapter:
        tools = await adapter.list_tools(use_cache=False)

    assert isinstance(tools, list)
    if tools:
        assert isinstance(tools[0], dict)
        assert "name" in tools[0]


@pytest.mark.asyncio
async def test_aenv_real_tool_call_if_configured(aenv_config):
    """Call a real tool when AENV_TOOL_NAME is configured."""
    tool_name = os.getenv("AENV_TOOL_NAME")
    if not tool_name:
        pytest.skip("Set AENV_TOOL_NAME to run real tool call integration test")

    tool_args = _load_tool_args_from_env()

    async with AenvEnvironmentAdapter(aenv_config) as adapter:
        result = await adapter.call_tool(tool_name=tool_name, arguments=tool_args)

    assert isinstance(result.is_error, bool)
    assert result.content is not None
