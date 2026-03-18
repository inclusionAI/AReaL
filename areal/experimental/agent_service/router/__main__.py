"""``python -m areal.experimental.agent_service.router``"""

import argparse

import uvicorn

from .app import create_router_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent Router")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    args = parser.parse_args()

    uvicorn.run(create_router_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
