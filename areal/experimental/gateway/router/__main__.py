"""CLI entrypoint for the router: ``python -m areal.experimental.gateway.router``."""

from __future__ import annotations

import argparse
import logging


def main():
    parser = argparse.ArgumentParser(description="AReaL Router")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=8081, help="Bind port")
    parser.add_argument(
        "--admin-api-key",
        default="areal-admin-key",
        help="Admin API key for privileged operations",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="Seconds between worker health polls",
    )
    parser.add_argument(
        "--worker-health-timeout",
        type=float,
        default=2.0,
        help="Timeout (seconds) per worker health check",
    )
    parser.add_argument(
        "--routing-strategy",
        default="round_robin",
        choices=["round_robin", "least_busy"],
        help="Worker selection strategy",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Log level",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="[Router] %(levelname)s %(message)s",
    )

    from areal.experimental.gateway.router.app import create_app
    from areal.experimental.gateway.router.config import RouterConfig

    config = RouterConfig(
        host=args.host,
        port=args.port,
        admin_api_key=args.admin_api_key,
        poll_interval=args.poll_interval,
        worker_health_timeout=args.worker_health_timeout,
        routing_strategy=args.routing_strategy,
        log_level=args.log_level,
    )

    import uvicorn

    app = create_app(config)
    uvicorn.run(app, host=config.host, port=config.port)


if __name__ == "__main__":
    main()
