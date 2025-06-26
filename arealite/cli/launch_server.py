# Copyright 2025 Ant Group Inc.
# Licensed under the Apache License, Version 2.0

import argparse
import os
from pathlib import Path

from hydra import compose as hydra_compose
from hydra import initialize as hydra_init
from omegaconf import OmegaConf

from arealite.api.cli_args import LLMServiceConfig
from arealite.api.llm_server_api import LLMServerFactory
from realhf.base import constants, logging, name_resolve, seeding

logger = logging.getLogger("Launch Server")


def main():
    """Main entry point for launching the LLM server."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", help="The path of the main configuration file", required=True
    )
    args, overrides = parser.parse_known_args()

    # Initialize hydra config
    config_file = Path(args.config).absolute()
    assert config_file.exists()
    relpath = Path(os.path.relpath(str(config_file), Path(__file__).parent.absolute()))
    hydra_init(config_path=str(relpath.parent), job_name="app", version_base=None)
    cfg = hydra_compose(
        config_name=str(relpath.name).rstrip(".yaml"),
        overrides=overrides,
    )

    # Merge with the default configuration
    default_cfg = OmegaConf.structured(LLMServiceConfig)
    cfg = OmegaConf.merge(default_cfg, cfg)
    cfg: LLMServiceConfig = OmegaConf.to_object(cfg)

    seeding.set_random_seed(cfg.seed, "llm_server")
    constants.set_experiment_trial_names(cfg.experiment_name, cfg.trial_name)
    name_resolve.reconfigure(cfg.cluster.name_resolve)

    server = LLMServerFactory.make_server(cfg)
    server.start()


if __name__ == "__main__":
    main()
