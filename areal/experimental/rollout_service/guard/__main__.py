"""CLI entrypoint: python -m areal.experimental.rollout_service.guard"""

from __future__ import annotations

import argparse
import os
import signal

from werkzeug.serving import make_server

from areal.api.cli_args import NameResolveConfig
from areal.experimental.rollout_service.guard import app as guard_app
from areal.experimental.rollout_service.guard.app import app as flask_app
from areal.utils import logging, name_resolve, names
from areal.utils.network import gethostip

logger = logging.getLogger("RPCGuard")


def main():
    """Main entry point for the RPCGuard service."""
    parser = argparse.ArgumentParser(
        description="AReaL RPCGuard — HTTP gateway for coordinating forked workers"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="Port to serve on (default: 0 = auto-assign)",
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)"
    )
    # name_resolve config
    parser.add_argument("--experiment-name", type=str, required=True)
    parser.add_argument("--trial-name", type=str, required=True)
    parser.add_argument("--role", type=str, required=True)
    parser.add_argument("--worker-index", type=int, default=-1)
    parser.add_argument("--name-resolve-type", type=str, default="nfs")
    parser.add_argument(
        "--nfs-record-root", type=str, default="/tmp/areal/name_resolve"
    )
    parser.add_argument("--etcd3-addr", type=str, default="localhost:2379")
    parser.add_argument(
        "--fileroot",
        type=str,
        default=None,
        help="Root directory for log files. If set, forked worker logs are written here.",
    )

    args, _ = parser.parse_known_args()

    # Set global config in app module for fork endpoint to use
    guard_app._server_host = args.host
    if guard_app._server_host == "0.0.0.0":
        guard_app._server_host = gethostip()

    guard_app._experiment_name = args.experiment_name
    guard_app._trial_name = args.trial_name
    guard_app._fileroot = args.fileroot

    # Get worker identity
    worker_role = args.role
    worker_index = args.worker_index
    if "SLURM_PROCID" in os.environ:
        # Overwriting with slurm task id
        worker_index = os.environ["SLURM_PROCID"]
    if worker_index == -1:
        raise ValueError("Invalid worker index. Not found from SLURM environ or args.")
    worker_id = f"{worker_role}/{worker_index}"

    # Make a flask server
    server = make_server(args.host, args.port, flask_app, threaded=True)
    server_port = server.socket.getsockname()[1]

    # Configure name_resolve
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
    name_resolve.add(key, f"{guard_app._server_host}:{server_port}", replace=True)

    logger.info(
        f"Starting RPCGuard on {guard_app._server_host}:{server_port} for worker {worker_id}"
    )

    def _sigterm_handler(signum, frame):
        """Convert SIGTERM to SystemExit so the finally block runs."""
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _sigterm_handler)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down RPCGuard (SIGINT)")
    except SystemExit:
        logger.info("Shutting down RPCGuard (SIGTERM)")
    finally:
        guard_app.cleanup_forked_children()
        server.shutdown()


if __name__ == "__main__":
    main()
