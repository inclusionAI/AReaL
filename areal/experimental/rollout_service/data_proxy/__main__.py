"""CLI entrypoint: python -m areal.experimental.rollout_service.data_proxy"""

from __future__ import annotations

import argparse
import os

import uvicorn

from areal.experimental.rollout_service.data_proxy.app import create_app
from areal.experimental.rollout_service.data_proxy.config import DataProxyConfig


def main():
    parser = argparse.ArgumentParser(description="AReaL Data Proxy")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument(
        "--backend-addr",
        default=os.environ.get("AREAL_DP_BACKEND_ADDR", "http://localhost:30000"),
    )
    parser.add_argument(
        "--tokenizer-path",
        default=os.environ.get("AREAL_DP_TOKENIZER_PATH"),
        required="AREAL_DP_TOKENIZER_PATH" not in os.environ,
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("AREAL_DP_LOG_LEVEL", "info"),
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=float(os.environ.get("AREAL_DP_REQUEST_TIMEOUT", "120.0")),
    )
    parser.add_argument(
        "--admin-api-key",
        default=os.environ.get("AREAL_DP_ADMIN_API_KEY", "areal-admin-key"),
    )
    args, _ = parser.parse_known_args()

    # Resolve the actual serving host (replace 0.0.0.0 with real IP)
    from areal.utils.network import gethostip

    serving_host = args.host
    if serving_host == "0.0.0.0":
        serving_host = gethostip()

    config = DataProxyConfig(
        host=args.host,
        port=args.port,
        backend_addr=args.backend_addr,
        tokenizer_path=args.tokenizer_path,
        log_level=args.log_level,
        request_timeout=args.request_timeout,
        admin_api_key=args.admin_api_key,
        serving_addr=f"{serving_host}:{args.port}",
    )
    app = create_app(config)
    uvicorn.run(app, host=config.host, port=config.port, log_level=config.log_level)


if __name__ == "__main__":
    main()
