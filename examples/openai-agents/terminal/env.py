import asyncio
import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any

import aiofiles
import aiofiles.os
import requests
from agents import FunctionTool, RunContextWrapper
from mcp import ClientSession
from mcp.client.sse import sse_client
from pydantic import BaseModel, Field

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
        dockerfile_contents: str | None = None,
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
        self.dockerfile_contents = dockerfile_contents

    async def _log_tool_call(
        self, tool_name: str, arguments: dict, result: str, container_name: str
    ):
        """Log tool call to file asynchronously.

        Args:
            tool_name: Name of the tool being called
            arguments: Arguments passed to the tool
            result: Result returned by the tool
            container_name: Name of the container for the log filename
        """
        if self.dump_dir is not None:
            dump_path = Path(self.dump_dir) / "terminal"
            await aiofiles.os.makedirs(dump_path, exist_ok=True)
            log_file = dump_path / f"{container_name}.jsonl"
            async with aiofiles.open(log_file, "a", encoding="utf-8") as f:
                log_entry = {
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "result": result,
                }
                await f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

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

    async def _call_mcp_tool_with_retry(self, tool_name: str, arguments: dict) -> str:
        """Call MCP tool with retry logic and error handling.

        Args:
            tool_name: Name of the MCP tool to call
            arguments: Arguments to pass to the tool

        Returns:
            Tool result as string

        Raises:
            RuntimeError: If tool call fails after all retries
        """
        for attempt in range(self.max_retries):
            try:
                result = await self._mcp_session.call_tool(
                    name=tool_name,
                    arguments=arguments,
                )
                output = (
                    result.content[0].text
                    if result.content and len(result.content) > 0
                    else ""
                )
                return output
            except Exception as e:
                logger.warning(f"Tool call attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    try:
                        await self._reconnect_mcp()
                    except Exception as reconnect_error:
                        logger.error(f"Reconnection failed: {reconnect_error}")
                        if attempt == self.max_retries - 2:
                            raise
                else:
                    raise RuntimeError(
                        f"Failed to call {tool_name} after {self.max_retries} attempts: {e}"
                    )

    def get_tools(self) -> list[FunctionTool]:
        """Create async function tools for terminal interaction.

        Returns:
            List of async function tools for keystrokes and capture-pane operations.
        """
        if not self.container_name:
            raise RuntimeError(
                "Container not started. Use TerminalEnv as context manager first."
            )

        def _format_terminal_output(output: str, keystrokes: str = "") -> str:
            """Clean terminal output by removing shell prompts and the exact keystrokes command.

            Args:
                output: Raw terminal output
                keystrokes: The exact command that was sent to remove from output
            """
            if not output:
                return output

            lines = output.split("\n")
            cleaned = []
            keystrokes_clean = keystrokes.strip()

            for line in lines:
                # Remove shell prompt
                line = re.sub(r"^[a-zA-Z0-9@]+:[^#]*#\s*", "", line)

                # Remove the exact keystrokes command if present at the start
                if keystrokes_clean and line.startswith(keystrokes_clean):
                    line = line[len(keystrokes_clean) :].lstrip()

                # Keep non-empty lines
                if line.strip():
                    cleaned.append(line.rstrip())

            return "\n".join(cleaned).strip()

        class ExecuteCommandArgs(BaseModel):
            """Arguments for executing a terminal command."""

            command: str = Field(
                description="The terminal command to execute (e.g., 'ls -la', 'python script.py', 'mkdir new_folder')"
            )
            wait_time_sec: float = Field(
                default=2.0,
                description="Time to wait after command execution in seconds. Use longer values for commands that take time to complete.",
            )

        class CurrentStatusArgs(BaseModel):
            """Arguments for getting current directory status and file listing."""

            include_hidden: bool = Field(
                default=False,
                description="Whether to include hidden files (starting with .) in the file listing. Set to True to see all files including dotfiles.",
            )

        class FileContentsArgs(BaseModel):
            """Arguments for viewing file contents with different viewing modes."""

            absolute_path: str = Field(
                description="Absolute path to the file to view (e.g., '/home/user/file.txt', '/var/log/syslog')"
            )
            tail_lines: int = Field(
                default=0,
                description="Number of lines to show from the end of the file. Use tail to see the last N lines (e.g., for logs). Set to 0 to disable tail view.",
            )
            head_lines: int = Field(
                default=0,
                description="Number of lines to show from the beginning of the file. Use head to see the first N lines (e.g., for file headers). Set to 0 to disable head view.",
            )

        async def execute_command(ctx: RunContextWrapper[Any], args: str) -> str:
            """Execute a command in the terminal and return the result.

            Args:
                args: JSON string containing command and wait_time_sec parameters.

            Returns:
                Command output after execution.
            """
            try:
                parsed_args = ExecuteCommandArgs.model_validate_json(args)

                # Call the simplified MCP wrapper
                raw_output = await self._call_mcp_tool_with_retry(
                    "keystrokes",
                    {
                        "container_name": self.container_name,
                        "keystrokes": parsed_args.command,
                        "append_enter": True,
                        "wait_time_sec": parsed_args.wait_time_sec,
                    },
                )

                # Clean the output
                output = _format_terminal_output(raw_output, parsed_args.command)

                # Log and track metrics
                await self._log_tool_call(
                    tool_name="execute_command",
                    arguments=parsed_args.model_dump(),
                    result=output,
                    container_name=self.container_name,
                )

                tracker = stats_tracker.get(self.rollout_stat_scope)
                tracker.scalar(
                    tool_execute_success=1.0,
                    tool_execute_input_chars=float(len(parsed_args.command)),
                    tool_execute_output_chars=float(len(output)),
                )

                return output
            except Exception as e:
                stats_tracker.get(self.rollout_stat_scope).scalar(
                    tool_execute_error=1.0
                )
                return f"Error executing command: {e}"

        async def current_working_directory(
            ctx: RunContextWrapper[Any], args: str
        ) -> str:
            """Get current working directory and list files.

            Args:
                args: JSON string containing include_hidden parameter.

            Returns:
                Current directory information and file listing.
            """
            try:
                parsed_args = CurrentStatusArgs.model_validate_json(args)

                # Get current working directory
                pwd_result = await self._call_mcp_tool_with_retry(
                    "keystrokes",
                    {
                        "container_name": self.container_name,
                        "keystrokes": "pwd",
                        "append_enter": True,
                        "wait_time_sec": 1.0,
                    },
                )

                # List files
                ls_option = "-alh" if parsed_args.include_hidden else "-lh"
                ls_result = await self._call_mcp_tool_with_retry(
                    "keystrokes",
                    {
                        "container_name": self.container_name,
                        "keystrokes": f"ls {ls_option}",
                        "append_enter": True,
                        "wait_time_sec": 1.0,
                    },
                )

                # Clean and combine results
                clean_pwd = _format_terminal_output(pwd_result, "pwd")
                clean_ls = _format_terminal_output(ls_result, f"ls {ls_option}")
                output = f"Current directory: {clean_pwd}\n\n Files: {clean_ls}"

                # Log and track metrics
                await self._log_tool_call(
                    tool_name="current_working_directory",
                    arguments=parsed_args.model_dump(),
                    result=output,
                    container_name=self.container_name,
                )

                tracker = stats_tracker.get(self.rollout_stat_scope)
                tracker.scalar(
                    tool_status_success=1.0,
                    tool_status_output_chars=float(len(output)),
                )

                return output
            except Exception as e:
                stats_tracker.get(self.rollout_stat_scope).scalar(tool_status_error=1.0)
                return f"Error getting current status: {e}"

        async def file_contents(ctx: RunContextWrapper[Any], args: str) -> str:
            try:
                parsed_args = FileContentsArgs.model_validate_json(args)

                # Determine the command to use based on arguments
                if parsed_args.tail_lines > 0:
                    command = (
                        f"tail -n {parsed_args.tail_lines} {parsed_args.absolute_path}"
                    )
                elif parsed_args.head_lines > 0:
                    command = (
                        f"head -n {parsed_args.head_lines} {parsed_args.absolute_path}"
                    )
                else:
                    command = f"cat {parsed_args.absolute_path}"

                # Get file contents
                raw_result = await self._call_mcp_tool_with_retry(
                    "keystrokes",
                    {
                        "container_name": self.container_name,
                        "keystrokes": command,
                        "append_enter": True,
                        "wait_time_sec": 1.0,
                    },
                )

                # Clean the output
                result = _format_terminal_output(raw_result, command)

                # Remove trailing shell prompt if present
                result = re.sub(r"\n[a-zA-Z0-9@]+:[^#]*#$", "", result)

                # Log and track metrics
                await self._log_tool_call(
                    tool_name="file_contents",
                    arguments=parsed_args.model_dump(),
                    result=result,
                    container_name=self.container_name,
                )

                tracker = stats_tracker.get(self.rollout_stat_scope)
                tracker.scalar(
                    tool_status_success=1.0,
                    tool_status_output_chars=float(len(result)),
                )

                return result
            except Exception as e:
                stats_tracker.get(self.rollout_stat_scope).scalar(tool_status_error=1.0)
                return f"Error getting file contents: {e}"

        return [
            FunctionTool(
                name="execute_command",
                description="Execute a command in the terminal and return the result.",
                params_json_schema=ExecuteCommandArgs.model_json_schema(),
                on_invoke_tool=execute_command,
            ),
            FunctionTool(
                name="current_working_directory",
                description="Get current working directory and list files.",
                params_json_schema=CurrentStatusArgs.model_json_schema(),
                on_invoke_tool=current_working_directory,
            ),
            FunctionTool(
                name="file_contents",
                description="View file contents with different viewing modes (full content, head, or tail).",
                params_json_schema=FileContentsArgs.model_json_schema(),
                on_invoke_tool=file_contents,
            ),
        ]

    def reward(self) -> float:
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
            return 0.0
