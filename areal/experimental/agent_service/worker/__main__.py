"""``python -m areal.experimental.agent_service.worker``"""

import argparse
import asyncio
import threading

import httpx
import uvicorn

from areal.utils.network import format_hostport

from .app import create_worker_app


def main() -> None:
    from ..data_proxy import create_data_proxy_app

    parser = argparse.ArgumentParser(description="Agent Worker + DataProxy")
    parser.add_argument("--agent", required=True, help="Agent import path")
    parser.add_argument("--router-addr", required=True, help="Router HTTP address")
    parser.add_argument("--worker-port", type=int, default=9000)
    parser.add_argument("--proxy-port", type=int, default=9100)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--admin-key", default="areal-agent-admin")
    args = parser.parse_args()

    worker_addr = f"http://{format_hostport(args.host, args.worker_port)}"
    proxy_addr = f"http://{format_hostport(args.host, args.proxy_port)}"

    worker_app = create_worker_app(args.agent)
    proxy_app = create_data_proxy_app(worker_addr=worker_addr)

    def run_worker():
        uvicorn.run(worker_app, host=args.host, port=args.worker_port, log_level="info")

    threading.Thread(target=run_worker, daemon=True).start()

    from ..auth import admin_headers

    async def register():
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{args.router_addr}/register",
                json={"addr": proxy_addr},
                headers=admin_headers(args.admin_key),
            )

    asyncio.run(register())
    uvicorn.run(proxy_app, host=args.host, port=args.proxy_port, log_level="info")


if __name__ == "__main__":
    main()
