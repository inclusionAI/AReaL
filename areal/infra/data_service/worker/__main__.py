# SPDX-License-Identifier: Apache-2.0

"""CLI entrypoint: python -m areal.infra.data_service.worker"""

from __future__ import annotations

import argparse
import importlib


def main():
    parser = argparse.ArgumentParser(description="AReaL Data Service Worker")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    args, _ = parser.parse_known_args()

    app_module = importlib.import_module("areal.infra.data_service.worker.app")
    config_module = importlib.import_module("areal.infra.data_service.worker.config")
    create_worker_app = getattr(app_module, "create_worker_app")
    DataWorkerConfig = getattr(config_module, "DataWorkerConfig")

    config = DataWorkerConfig(
        host=args.host,
        port=args.port,
        rank=args.rank,
        world_size=args.world_size,
        dataloader_num_workers=args.dataloader_num_workers,
    )
    uvicorn = importlib.import_module("uvicorn")

    app = create_worker_app(config)
    uvicorn.run(app, host=config.host, port=config.port, log_level="warning")


if __name__ == "__main__":
    main()
