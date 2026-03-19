"""``python -m areal.experimental.agent_service.data_proxy``"""

from .app import create_data_proxy_app

if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Agent DataProxy")
    parser.add_argument("--worker-addr", required=True, help="Worker HTTP address")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9100)
    args = parser.parse_args()

    uvicorn.run(
        create_data_proxy_app(worker_addr=args.worker_addr),
        host=args.host,
        port=args.port,
    )
