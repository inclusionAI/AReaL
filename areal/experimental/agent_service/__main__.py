"""Agent Service entry point.

Start the Agent Service with a Gateway and Agent Workers managed
via the Scheduler API.

Examples::

    # Start with 2 workers
    python -m areal.experimental.agent_service --workers 2 --agent-import-path your.Agent
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from areal.utils import logging
from areal.utils.network import find_free_ports, gethostip

logger = logging.getLogger("AgentService")


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Agent Service")

    # Server config
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8300)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of Agent Worker processes (default: 1)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory for log files.",
    )

    # Agent config
    parser.add_argument(
        "--agent-import-path",
        type=str,
        default=None,
        help="Agent class import path",
    )
    parser.add_argument(
        "--agent-reuse",
        action="store_true",
        help="Reuse a single agent instance",
    )
    parser.add_argument(
        "--agent-init-kwargs",
        type=str,
        default=None,
        help="JSON-encoded agent init kwargs",
    )

    # Scheduler config
    parser.add_argument(
        "--scheduler-type",
        type=str,
        choices=["local", "ray", "slurm"],
        default="local",
        help="Scheduler backend (default: local)",
    )

    return parser.parse_args()


def _get_agent_config(
    args: argparse.Namespace,
) -> tuple[str | None, bool, dict[str, Any]]:
    """Get agent configuration from CLI args.

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
    # Read directly from CLI args
    agent_import_path = args.agent_import_path
    # Note: agent_import_path can be None for dynamic mode

    # agent_reuse: CLI flag only
    agent_reuse = args.agent_reuse

    # Shared mode requires agent_import_path at startup
    if agent_reuse and not agent_import_path:
        raise ValueError(
            "agent_import_path required for shared mode (agent_reuse=True). "
            "Use --agent-import-path."
        )

    # agent_init_kwargs: CLI arg only
    agent_init_kwargs: dict[str, Any] = {}
    kwargs_str = args.agent_init_kwargs or ""
    if kwargs_str:
        try:
            agent_init_kwargs = json.loads(kwargs_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON for agent_init_kwargs: {e}") from e

    return agent_import_path, agent_reuse, agent_init_kwargs


def main() -> None:
    """Start the Agent Service."""
    import signal

    from areal.api.cli_args import AgentServiceSpec
    from areal.infra.scheduler.local import LocalScheduler

    from .agent_controller import AgentController
    from .config import GatewayConfig

    args = _parse_args()
    agent_import_path, agent_reuse, agent_init_kwargs = _get_agent_config(args)
    host = gethostip() if args.host == "0.0.0.0" else args.host
    port = args.port if args.port != 0 else find_free_ports(1)[0]
    workers = args.workers

    # Create scheduler
    scheduler_type = args.scheduler_type
    if scheduler_type == "local":
        log_dir = args.log_dir or "/tmp/areal/agent-service/logs"
        scheduler = LocalScheduler(log_dir=log_dir)
    elif scheduler_type == "ray":
        raise NotImplementedError(
            "Ray scheduler requires exp_config; use Trainer integration"
        )
    elif scheduler_type == "slurm":
        raise NotImplementedError(
            "Slurm scheduler requires cluster config; use Trainer integration"
        )

    config = GatewayConfig(
        host=host,
        port=port,
    )
    controller = AgentController(config=config, scheduler=scheduler)

    spec = AgentServiceSpec(
        agent_import_path=agent_import_path or "",
        agent_reuse=agent_reuse,
        agent_init_kwargs=agent_init_kwargs,
        workers=workers,
    )
    gateway_addr = controller.start(spec)

    logger.info(
        "Agent Service started at %s with %d worker(s). Press Ctrl+C to stop.",
        gateway_addr,
        workers,
    )

    try:
        signal.pause()
    except (KeyboardInterrupt, AttributeError):
        # AttributeError: signal.pause() not available on Windows
        import time

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    finally:
        logger.info("Shutting down Agent Service")
        controller.stop()


if __name__ == "__main__":
    main()
