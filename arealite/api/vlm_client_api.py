# Copyright 2025 Ant Group Inc.
# Licensed under the Apache License, Version 2.0

import abc
import asyncio
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import aiohttp
import requests
import transformers

from arealite.api.cli_args import LLMClientConfig, TrainingArgs
from arealite.api.io_struct import (
    LLMRequest,
    LLMResponse,
    LLMServerInfo,
    WeightMeta,
    WeightUpdateGroupMeta,
)
from arealite.api.llm_server_api import LLMServiceRegistry
from realhf.api.core.data_api import load_hf_processor_and_tokenizer
from arealite.api.llm_client_api import LLMClient 

class VLMClient(LLMClient):
    """A client for interacting with VLM servers."""

    def __init__(self, args: TrainingArgs, client_config: LLMClientConfig):
        super().__init__(args, client_config)
        self.registry = LLMServiceRegistry(args.experiment_name, args.trial_name)
        self.processor, self.tokenizer = load_hf_processor_and_tokenizer(
            args.rollout.model_path
        )

