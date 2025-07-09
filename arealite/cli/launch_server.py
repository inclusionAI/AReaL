# Copyright 2025 Ant Group Inc.
# Licensed under the Apache License, Version 2.0

import sys

from arealite.api.cli_args import TrainingArgs, prepare_training_args
from arealite.api.llm_server_api import LLMServerFactory
from realhf.base import seeding


def main():
    """Main entry point for launching the LLM server."""
    cfg: TrainingArgs = prepare_training_args(sys.argv[1:])[0]
    seeding.set_random_seed(cfg.seed, "llm_server")
    server = LLMServerFactory(cfg).make_server(cfg.rollout.llm_service)
    server.start()


if __name__ == "__main__":
    main()
