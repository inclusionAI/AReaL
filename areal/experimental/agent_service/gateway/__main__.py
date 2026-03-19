"""``python -m areal.experimental.agent_service.gateway``"""

import argparse

import uvicorn

from .app import create_gateway_app
from .bridge import OpenResponsesBridge, mount_bridge


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent Gateway")
    parser.add_argument("--router-addr", required=True, help="Router HTTP address")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--admin-key", default="areal-agent-admin")
    args = parser.parse_args()

    app = create_gateway_app(router_addr=args.router_addr, admin_key=args.admin_key)
    mount_bridge(
        app,
        OpenResponsesBridge(router_addr=args.router_addr, admin_key=args.admin_key),
        admin_key=args.admin_key,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
