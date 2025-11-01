import asyncio
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any

import requests
from agents import FunctionTool, RunContextWrapper
from mcp import ClientSession
from mcp.client.sse import sse_client
from pydantic import BaseModel

from areal.utils import stats_tracker

logger = logging.getLogger(__name__)


class TerminalEnv:
    """Simple context manager for managing terminal container lifecycle."""

    def __init__(
        self,
        task_name: str = None,
        dump_dir: str | None = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        rollout_stat_scope: str = "rollout",
    ):
        self.base_url = os.environ.get("MCP_SERVER_URL", "http://localhost:8000")
        self.task_name = task_name
        self.container_name = None
        self.uuid = str(uuid.uuid4())
        self.dump_dir = dump_dir
        self._mcp_session = None
        self._sse_exit_stack = None
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._connection_lock = asyncio.Lock()
        self.rollout_stat_scope = rollout_stat_scope

    def __enter__(self) -> "TerminalEnv":
        """Start the task container."""
        payload = {"uuid": self.uuid, "task_name": self.task_name}
        try:
            response = requests.post(
                f"{self.base_url}/tasks/start", json=payload, timeout=360
            )
            response.raise_for_status()
            data = response.json()
            self.container_name = data["container_name"]
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to start task container: {e}")
        return self

    async def _connect_mcp(self) -> None:
        """Establish MCP connection with retry logic."""
        from contextlib import AsyncExitStack

        for attempt in range(self.max_retries):
            try:
                if self._sse_exit_stack is None:
                    self._sse_exit_stack = AsyncExitStack()

                logger.info(
                    f"Connecting to MCP server (attempt {attempt + 1}/{self.max_retries})..."
                )
                read, write = await self._sse_exit_stack.enter_async_context(
                    sse_client(
                        url=f"{self.base_url}/sse",
                        timeout=60 * 3,
                    )
                )
                self._mcp_session = await self._sse_exit_stack.enter_async_context(
                    ClientSession(read, write)
                )
                await self._mcp_session.initialize()
                logger.info("MCP connection established successfully")
                return
            except Exception as e:
                logger.warning(f"MCP connection attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(
                        self.retry_delay * (attempt + 1)
                    )  # Exponential backoff
                    # Clean up failed connection attempt
                    if self._sse_exit_stack:
                        try:
                            await self._sse_exit_stack.aclose()
                        except Exception:
                            pass
                        self._sse_exit_stack = None
                    self._mcp_session = None
                else:
                    raise RuntimeError(
                        f"Failed to connect to MCP server after {self.max_retries} attempts: {e}"
                    )

    async def _reconnect_mcp(self) -> None:
        """Reconnect to MCP server after connection loss."""
        async with self._connection_lock:
            logger.info("Reconnecting to MCP server...")
            # Clean up old connection
            if self._sse_exit_stack:
                try:
                    await self._sse_exit_stack.aclose()
                except Exception as e:
                    logger.warning(f"Error closing old connection: {e}")
                self._sse_exit_stack = None
            self._mcp_session = None

            # Establish new connection
            await self._connect_mcp()

    async def __aenter__(self) -> "TerminalEnv":
        """Async context manager entry - start container and MCP session."""
        self.__enter__()
        # Initialize persistent MCP connection with retry
        await self._connect_mcp()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the task container on exit."""
        if self.container_name:
            try:
                payload = {"uuid": self.uuid, "task_name": self.task_name}
                response = requests.post(
                    f"{self.base_url}/tasks/stop", json=payload, timeout=30
                )
                response.raise_for_status()
            except Exception as e:
                print(f"Warning: Failed to stop task {self.task_name}: {e}")
        return False  # Don't suppress exceptions

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup MCP session and container."""
        if self._sse_exit_stack:
            await self._sse_exit_stack.aclose()
        self.__exit__(exc_type, exc_val, exc_tb)
        return False

    def get_tools(self) -> list[FunctionTool]:
        """Create async function tools for terminal interaction.

        Returns:
            List of async function tools for keystrokes and capture-pane operations.
        """
        if not self.container_name:
            raise RuntimeError(
                "Container not started. Use TerminalEnv as context manager first."
            )

        # Store container_name and base_url in closure
        container_name = self.container_name

        class SendKeystrokesArgs(BaseModel):
            keystrokes: str
            append_enter: bool = False
            wait_time_sec: float = 0.0

        class CapturePaneArgs(BaseModel):
            wait_before_capture_sec: float = 0.0

        async def send_keystrokes(ctx: RunContextWrapper[Any], args: str) -> str:
            """Send keystrokes to the terminal.

            Args:
                args: JSON string containing keystrokes, append_enter, and wait_time_sec parameters.

            Returns:
                Terminal output after executing the keystrokes.
            """
            try:
                parsed_args = SendKeystrokesArgs.model_validate_json(args)
                tracker = stats_tracker.get(self.rollout_stat_scope)
                retries = 0.0
                with tracker.record_timing("tool/keystrokes"):
                    for attempt in range(self.max_retries):
                        try:
                            result = await self._mcp_session.call_tool(
                                name="keystrokes",
                                arguments={
                                    "container_name": container_name,
                                    "keystrokes": parsed_args.keystrokes,
                                    "append_enter": parsed_args.append_enter,
                                    "wait_time_sec": parsed_args.wait_time_sec,
                                },
                            )
                            break
                        except Exception as e:
                            logger.warning(
                                f"Tool call attempt {attempt + 1} failed: {e}"
                            )
                            if attempt < self.max_retries - 1:
                                retries += 1.0
                                try:
                                    await self._reconnect_mcp()
                                except Exception as reconnect_error:
                                    logger.error(
                                        f"Reconnection failed: {reconnect_error}"
                                    )
                                    if attempt == self.max_retries - 2:
                                        raise
                            else:
                                raise

                output = (
                    result.content[0].text
                    if result.content and len(result.content) > 0
                    else ""
                )

                if self.dump_dir is not None:
                    dump_path = Path(self.dump_dir) / "terminal"
                    dump_path.mkdir(parents=True, exist_ok=True)
                    log_file = dump_path / f"{container_name}.jsonl"
                    with open(log_file, "a", encoding="utf-8") as f:
                        log_entry = {
                            "tool_name": "keystrokes",
                            "arguments": parsed_args.model_dump(),
                            "result": output,
                        }
                        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

                tracker.scalar(
                    tool_keystrokes_success=1.0,
                    tool_keystrokes_retries=retries,
                    tool_keystrokes_input_chars=float(len(parsed_args.keystrokes)),
                    tool_keystrokes_output_chars=float(len(output)),
                )
                return output
            except Exception as e:
                stats_tracker.get(self.rollout_stat_scope).scalar(
                    tool_keystrokes_error=1.0
                )
                return f"Error sending keystrokes: {e}"

        async def capture_pane(ctx: RunContextWrapper[Any], args: str) -> str:
            """Capture the current terminal pane output.

            Args:
                args: JSON string containing wait_before_capture_sec parameter.

            Returns:
                Current terminal pane content.
            """

            try:
                parsed_args = CapturePaneArgs.model_validate_json(args)
                tracker = stats_tracker.get(self.rollout_stat_scope)
                retries = 0.0
                with tracker.record_timing("tool/capture_pane"):
                    for attempt in range(self.max_retries):
                        try:
                            result = await self._mcp_session.call_tool(
                                name="capture-pane",
                                arguments={
                                    "container_name": container_name,
                                    "wait_before_capture_sec": parsed_args.wait_before_capture_sec,
                                },
                            )
                            break
                        except Exception as e:
                            logger.warning(
                                f"Tool call attempt {attempt + 1} failed: {e}"
                            )
                            if attempt < self.max_retries - 1:
                                retries += 1.0
                                try:
                                    await self._reconnect_mcp()
                                except Exception as reconnect_error:
                                    logger.error(
                                        f"Reconnection failed: {reconnect_error}"
                                    )
                                    if attempt == self.max_retries - 2:
                                        raise
                            else:
                                raise

                output = (
                    result.content[0].text
                    if result.content and len(result.content) > 0
                    else ""
                )

                if self.dump_dir is not None:
                    dump_path = Path(self.dump_dir) / "terminal"
                    dump_path.mkdir(parents=True, exist_ok=True)
                    log_file = dump_path / f"{container_name}.jsonl"
                    with open(log_file, "a", encoding="utf-8") as f:
                        log_entry = {
                            "tool_name": "capture_pane",
                            "arguments": parsed_args.model_dump(),
                            "result": output,
                        }
                        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

                tracker.scalar(
                    tool_capture_success=1.0,
                    tool_capture_retries=retries,
                    tool_capture_output_chars=float(len(output)),
                )
                return output
            except Exception as e:
                stats_tracker.get(self.rollout_stat_scope).scalar(
                    tool_capture_error=1.0
                )
                return f"Error capturing pane: {e}"

        return [
            FunctionTool(
                name="send_keystrokes",
                description="Send keystrokes to the terminal.",
                params_json_schema=SendKeystrokesArgs.model_json_schema(),
                on_invoke_tool=send_keystrokes,
            ),
            FunctionTool(
                name="capture_pane",
                description="Capture the current terminal pane output.",
                params_json_schema=CapturePaneArgs.model_json_schema(),
                on_invoke_tool=capture_pane,
            ),
        ]

    def reward(self) -> float | None:
        """Reward function for the terminal environment."""
        try:
            payload = {"container_name": self.container_name}
            response = requests.post(
                f"{self.base_url}/tasks/validate", json=payload, timeout=360
            )
            response.raise_for_status()
            result = response.json()
            return round(float(result["score"]), 2)
        except Exception as e:
            print(f"Error getting reward from API: {e}")
            return None
