"""CLI entrypoint: python -m areal.experimental.gateway.data_proxy"""

from __future__ import annotations

import argparse

import uvicorn

from areal.experimental.gateway.data_proxy.app import create_app
from areal.experimental.gateway.data_proxy.config import DataProxyConfig


def main():
    parser = argparse.ArgumentParser(description="AReaL Data Proxy")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--backend-addr", default="http://localhost:30000")
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--log-level", default="info")
    parser.add_argument("--request-timeout", type=float, default=120.0)
    args = parser.parse_args()

    config = DataProxyConfig(
        host=args.host,
        port=args.port,
        backend_addr=args.backend_addr,
        tokenizer_path=args.tokenizer_path,
        log_level=args.log_level,
        request_timeout=args.request_timeout,
    )
    app = create_app(config)
    uvicorn.run(app, host=config.host, port=config.port, log_level=config.log_level)


if __name__ == "__main__":
    main()
