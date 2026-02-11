"""Agent Service entry point.

This module provides the command-line entry point for running the Agent Service
as a standalone process. The Agent Service can be started via:

    python -m areal.experimental.agent_service --experiment-name exp --trial-name trial --role agent --worker-index 0

The Agent Service runs agent.run() in an independent process, accepting HTTP
requests from OpenAIProxyWorkflow (mode="service").

Environment Variables
---------------------
AGENT_IMPORT_PATH : str (required)
    Import path for the agent class (e.g., "mymodule.MyAgent").
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import uvicorn

from areal.api.cli_args import NameResolveConfig
from areal.utils import logging, name_resolve, names
from areal.utils.network import find_free_ports, gethostip

from .rpc_server import (
    ENV_AGENT_HOST,
    ENV_AGENT_IMPORT_PATH_INTERNAL,
    ENV_AGENT_INIT_KWARGS_INTERNAL,
    ENV_AGENT_PORT,
    ENV_AGENT_REUSE_INTERNAL,
    ENV_AGENT_WORKERS,
)

logger = logging.getLogger("AgentRPCServer")


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Agent RPC Server")

    # Server config
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8300)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)",
    )

    # Agent config (CLI args, override environment variables)
    parser.add_argument(
        "--agent-import-path",
        type=str,
        default=None,
        help="Agent class import path (overrides AGENT_IMPORT_PATH)",
    )
    parser.add_argument(
        "--agent-reuse",
        action="store_true",
        help="Reuse a single agent instance (overrides AGENT_REUSE)",
    )
    parser.add_argument(
        "--agent-init-kwargs",
        type=str,
        default=None,
        help="JSON-encoded agent init kwargs (overrides AGENT_INIT_KWARGS)",
    )

    # Framework integration (optional for standalone mode)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--trial-name", type=str, default=None)
    parser.add_argument("--role", type=str, default=None)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--name-resolve-type", type=str, default="nfs")
    parser.add_argument(
        "--nfs-record-root", type=str, default="/tmp/areal/name_resolve"
    )
    parser.add_argument("--etcd3-addr", type=str, default="localhost:2379")
    parser.add_argument(
        "--fileroot",
        type=str,
        default=None,
        help="Root directory for log files (unused, for compatibility)",
    )
    return parser.parse_known_args()[0]


def _get_agent_config(
    args: argparse.Namespace,
) -> tuple[str | None, bool, dict[str, Any]]:
    """Get agent configuration from CLI args (priority) or environment variables.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments.

    Returns
    -------
    tuple[str | None, bool, dict[str, Any]]
        (agent_import_path, agent_reuse, agent_init_kwargs)
        agent_import_path can be None for dynamic mode (per-request selection).

    Raises
    ------
    ValueError
        If agent_reuse=True but agent_import_path is not set,
        or if agent_init_kwargs is invalid JSON.
    """
    # CLI args take priority over environment variables
    agent_import_path = args.agent_import_path or os.environ.get("AGENT_IMPORT_PATH")
    # Note: agent_import_path can be None for dynamic mode

    # agent_reuse: CLI flag or environment variable
    agent_reuse = args.agent_reuse
    if not agent_reuse:
        agent_reuse_str = os.environ.get("AGENT_REUSE", "false").lower()
        agent_reuse = agent_reuse_str in ("true", "1")

    # Shared mode requires agent_import_path at startup
    if agent_reuse and not agent_import_path:
        raise ValueError(
            "agent_import_path required for shared mode (agent_reuse=True). "
            "Use --agent-import-path or AGENT_IMPORT_PATH environment variable."
        )

    # agent_init_kwargs: CLI arg takes priority
    agent_init_kwargs: dict[str, Any] = {}
    kwargs_str = args.agent_init_kwargs or os.environ.get("AGENT_INIT_KWARGS", "")
    if kwargs_str:
        try:
            agent_init_kwargs = json.loads(kwargs_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON for agent_init_kwargs: {e}") from e

    return agent_import_path, agent_reuse, agent_init_kwargs


def _resolve_worker_index(args: argparse.Namespace) -> int:
    """Resolve worker index from args or SLURM environment.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments.

    Returns
    -------
    int
        The resolved worker index.

    Raises
    ------
    ValueError
        If worker index cannot be determined.
    """
    worker_index = args.worker_index
    if "SLURM_PROCID" in os.environ:
        worker_index = int(os.environ["SLURM_PROCID"])
    if worker_index == -1:
        raise ValueError("Invalid worker index. Not found from SLURM environ or args.")
    return worker_index


def _set_config_env_vars(
    host: str,
    port: int,
    workers: int,
    agent_import_path: str | None,
    agent_reuse: bool,
    agent_init_kwargs: dict[str, Any],
) -> None:
    """Set environment variables for multi-worker mode.

    These environment variables are set in the main process before forking
    and are inherited by all worker processes.

    Parameters
    ----------
    host : str
        Host address for the server.
    port : int
        Port number for the server.
    workers : int
        Number of worker processes.
    agent_import_path : str | None
        Import path for the agent class.
    agent_reuse : bool
        Whether to reuse agent instances.
    agent_init_kwargs : dict[str, Any]
        Keyword arguments for agent initialization.
    """
    os.environ[ENV_AGENT_HOST] = host
    os.environ[ENV_AGENT_PORT] = str(port)
    os.environ[ENV_AGENT_WORKERS] = str(workers)
    os.environ[ENV_AGENT_IMPORT_PATH_INTERNAL] = agent_import_path or ""
    os.environ[ENV_AGENT_REUSE_INTERNAL] = "true" if agent_reuse else "false"
    os.environ[ENV_AGENT_INIT_KWARGS_INTERNAL] = (
        json.dumps(agent_init_kwargs) if agent_init_kwargs else ""
    )


def main() -> None:
    """Main entry point for the Agent RPC server."""
    args = _parse_args()
    agent_import_path, agent_reuse, agent_init_kwargs = _get_agent_config(args)

    # Resolve host and port
    host = gethostip() if args.host == "0.0.0.0" else args.host
    port = args.port if args.port != 0 else find_free_ports(1)[0]
    workers = args.workers

    # Determine if running in standalone mode (no experiment-name means standalone)
    is_standalone = args.experiment_name is None

    mode_str = "shared" if agent_reuse else "per-request"
    path_str = agent_import_path or "(dynamic)"
    workers_str = f", workers={workers}" if workers > 1 else ""

    if not is_standalone:
        # Framework mode: configure name_resolve and register
        worker_index = _resolve_worker_index(args)
        name_resolve.reconfigure(
            NameResolveConfig(
                type=args.name_resolve_type,
                nfs_record_root=args.nfs_record_root,
                etcd3_addr=args.etcd3_addr,
            )
        )
        key = names.worker_discovery(
            args.experiment_name, args.trial_name, args.role, worker_index
        )
        name_resolve.add(key, f"{host}:{port}", replace=True)
        logger.info(
            f"Starting Agent RPC server on {host}:{port} "
            f"for worker {args.role}/{worker_index}, "
            f"agent_import_path={path_str}, mode={mode_str}{workers_str}"
        )
    else:
        # Standalone mode: skip name_resolve registration
        logger.info(
            f"Starting Agent RPC server in standalone mode on {host}:{port}, "
            f"agent_import_path={path_str}, mode={mode_str}{workers_str}"
        )

    try:
        # Set env vars unconditionally (both single and multi-worker use create_app)
        _set_config_env_vars(
            host, port, workers, agent_import_path, agent_reuse, agent_init_kwargs
        )
        uvicorn.run(
            "areal.experimental.agent_service.rpc_server:create_app",
            factory=True,
            host="0.0.0.0",
            port=port,
            workers=workers,
            log_level="warning",
            access_log=False,
        )
    except KeyboardInterrupt:
        logger.info("Shutting down Agent RPC server")
    finally:
        logger.info("Agent RPC server stopped.")


if __name__ == "__main__":
    main()
