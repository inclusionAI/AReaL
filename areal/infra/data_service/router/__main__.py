from __future__ import annotations

import argparse
import importlib


def main():
    parser = argparse.ArgumentParser(description="AReaL Data Service Router")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--admin-api-key", default="areal-data-admin")
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--worker-health-timeout", type=float, default=3.0)
    args, _ = parser.parse_known_args()

    router_app_module = importlib.import_module("areal.infra.data_service.router.app")
    router_config_module = importlib.import_module(
        "areal.infra.data_service.router.config"
    )
    create_router_app = router_app_module.create_router_app
    RouterConfig = router_config_module.RouterConfig

    config = RouterConfig(
        host=args.host,
        port=args.port,
        admin_api_key=args.admin_api_key,
        poll_interval=args.poll_interval,
        worker_health_timeout=args.worker_health_timeout,
    )

    uvicorn = importlib.import_module("uvicorn")

    app = create_router_app(config)
    uvicorn.run(app, host=config.host, port=config.port, log_level="warning")


if __name__ == "__main__":
    main()
