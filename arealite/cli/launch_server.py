# Copyright 2025 Ant Group Inc.
# Licensed under the Apache License, Version 2.0

import hydra
from omegaconf import OmegaConf

from arealite.api.cli_args import LLMServiceConfig
from arealite.api.llm_server_api import LLMServerFactory
from realhf.base import constants, name_resolve, seeding


@hydra.main(version_base=None, config_path="config", config_name="llm_server")
def main(args: LLMServiceConfig):
    """Main entry point for launching the LLM server."""
    default_args = OmegaConf.structured(LLMServiceConfig)
    args = OmegaConf.merge(default_args, args)
    args: LLMServiceConfig = OmegaConf.to_object(args)

    seeding.set_random_seed(args.seed, "llm_server")

    constants.set_experiment_trial_names(args.experiment_name, args.trial_name)
    name_resolve.reconfigure(args.cluster.name_resolve)

    server = LLMServerFactory.make_server(args)
    server.start()


if __name__ == "__main__":
    main()
