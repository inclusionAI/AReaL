"""A MCP server for running terminal-bench tasks independently."""

import argparse
import asyncio
import contextvars
import logging
import warnings
from pathlib import Path

from fastmcp import FastMCP
from starlette.requests import Request

from .logging_config import setup_logging
from .models import TaskRequest, TerminalBenchServer

# Suppress SQLAlchemy 2.0 deprecation warnings from terminal_bench
warnings.filterwarnings("ignore", category=DeprecationWarning, module="terminal_bench")

setup_logging()

logger = logging.getLogger(__name__)

# Context variable to track request start time
request_start_time = contextvars.ContextVar("request_start_time", default=None)

server_instance = None
# --- MCP Server Setup with FastMCP ---

mcp = FastMCP("t-bench-multi-task")


@mcp.tool()
async def keystrokes(
    container_name: str,
    keystrokes: str,
    append_enter: bool = False,
    wait_time_sec: float = 0.0,
) -> str:
    """Send keystrokes to a tmux session and return the result.

    Args:
        container_name: The name of the container to send keystrokes to
        keystrokes: Keystrokes to execute in the terminal. Use tmux-style escape sequences for special characters (e.g. C-c for ctrl-c)
        append_enter: Whether to append a newline character to the end of the keystrokes (necessary to execute bash commands)
        wait_time_sec: The number of expected seconds to wait for the command to complete

    Returns:
        Terminal output after executing the keystrokes
    """

    # Validate container exists
    if not server_instance.validate_container(container_name):
        raise ValueError(f"Invalid or unknown container_name: {container_name}")

    # Update last seen timestamp
    server_instance.update_task_last_seen(container_name)

    # Get or create tmux session
    session = await server_instance.get_tmux_session(container_name)

    # Clear the terminal to avoid historical results in next calls
    session.send_keys(
        keys=["clear", "Enter"],
        min_timeout_sec=0.1,
        max_timeout_sec=0.1,
    )

    keys = [keystrokes, "Enter"] if append_enter else keystrokes
    session.send_keys(
        keys=keys,
        min_timeout_sec=wait_time_sec,
        max_timeout_sec=wait_time_sec,
    )

    # Capture the output before clearing
    output = session.capture_pane()

    # Clear the terminal to avoid historical results in next calls
    session.send_keys(
        keys=["clear", "Enter"],
        min_timeout_sec=0.1,
        max_timeout_sec=0.1,
    )

    return output


@mcp.tool()
async def capture_pane(
    container_name: str,
    wait_before_capture_sec: float = 0.0,
) -> str:
    """Capture the pane of a tmux session.

    Args:
        container_name: The name of the container to capture the pane from
        wait_before_capture_sec: The number of seconds to wait before capturing the pane. This is useful if you just executed a command and want to wait a bit to capture the output

    Returns:
        Current terminal pane content
    """

    # Validate container exists
    if not server_instance.validate_container(container_name):
        raise ValueError(f"Invalid or unknown container_name: {container_name}")

    # Update last seen timestamp
    server_instance.update_task_last_seen(container_name)

    # Get or create tmux session
    session = await server_instance.get_tmux_session(container_name)

    if wait_before_capture_sec > 0:
        await asyncio.sleep(wait_before_capture_sec)

    return session.capture_pane()


# --- Custom HTTP Routes for Task Management ---


@mcp.custom_route("/tasks/start", methods=["POST"])
async def start_task_route(request: Request):
    """Start a new task container."""
    data = await request.json()
    req = TaskRequest.model_validate(data)
    return await server_instance.start_task(req)


@mcp.custom_route("/tasks/stop", methods=["POST"])
async def stop_task_route(request: Request):
    """Stop a task container."""
    data = await request.json()
    req = TaskRequest.model_validate(data)
    return await server_instance.stop_task(req)


@mcp.custom_route("/tasks", methods=["GET"])
async def list_tasks_route(request: Request):
    """List all active tasks."""
    return await server_instance.list_tasks()


@mcp.custom_route("/tasks/validate", methods=["POST"])
async def validate_container_route(request: Request):
    """Validate a container and get test results."""
    data = await request.json()
    req = TaskRequest.model_validate(data)
    container_name = req.container_name
    if not container_name or container_name == "":
        container_name = f"{req.uuid}__{req.task_name}"
    return await server_instance.validate_task(container_name)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Terminal Bench MCP Server - Run terminal-bench tasks independently",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--tasks-dir",
        type=Path,
        default=None,
        help="Path to the tasks directory (default: env T_BENCH_TASKS_DIR or /app/tasks)",
    )

    parser.add_argument(
        "--tasks-log-dir",
        type=Path,
        default=None,
        help="Path to the tasks logs directory (default: env T_BENCH_TASKS_LOG_DIR or /var/logs/terminal-bench/)",
    )

    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to bind the server to"
    )

    parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind the server to"
    )

    parser.add_argument(
        "--preheat-image",
        action="store_true",
        default=False,
        help="Preheat the docker image before starting the server",
    )

    return parser.parse_args()


def main():
    """Main entry point for the server."""
    args = parse_args()

    # Initialize server instance (don't start yet)
    global server_instance
    server_instance = TerminalBenchServer(
        tasks_dir=args.tasks_dir,
        tasks_log_dir=args.tasks_log_dir,
        preheat_image=args.preheat_image,
    )

    server_instance.startup()

    # Run FastMCP server - startup will happen on first tool call
    try:
        mcp.run(
            transport="sse",
            host=args.host,
            port=args.port,
        )
    except KeyboardInterrupt:
        print("\nShutting down server...")
        if server_instance:
            server_instance.shutdown()


if __name__ == "__main__":
    main()
