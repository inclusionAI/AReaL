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
from realhf.api.core.data_api import load_hf_tokenizer


class LLMClient(abc.ABC):
    def __init__(self, args: TrainingArgs, client_config: LLMClientConfig):
        self.args = args
        self.client_config = client_config

        self.registry = LLMServiceRegistry(args.experiment_name, args.trial_name)
        self.tokenizer: transformers.PreTrainedTokenizerFast = load_hf_tokenizer(
            args.rollout.model_path
        )

    def select_server(self):
        """Get an available healthy server."""
        servers = self.get_healthy_servers()
        min_load = min([server.load for server in servers])
        servers = [server for server in servers if server.load == min_load]
        return random.choice(servers)

    def get_healthy_servers(self):
        servers = self.registry.get_healthy_servers()
        if not servers:
            raise RuntimeError("No healthy SGLang servers available")
        return servers

    def wait_until_servers_ready(self):
        while len(self.registry.get_healthy_servers()) == 0:
            time.sleep(10)

    async def arequest_with_retry(
        self,
        endpoint: str,
        payload: Optional[Dict[str, Any]] = None,
        method: str = "POST",
        max_retries: Optional[int] = None,
        timeout: Optional[float] = None,
        retry_delay: float = 1.0,
        target_server: Optional[LLMServerInfo] = None,
    ) -> tuple[aiohttp.ClientResponse, LLMServerInfo]:
        timeout = timeout or self.client_config.request_timeout
        last_exception = None
        max_retries = max_retries or self.client_config.request_retries

        # Try with retries
        for _ in range(max_retries):
            if target_server is None:
                server_info = self.select_server()
            else:
                server_info = target_server
            base_url = f"http://{server_info.host}:{server_info.port}"
            url = f"{base_url}{endpoint}"

            for attempt in range(max_retries):
                try:
                    async with aiohttp.ClientSession(
                        timeout=aiohttp.ClientTimeout(
                            total=timeout,
                            sock_connect=30,
                            sock_read=timeout,
                        )
                    ) as session:
                        if method.upper() == "GET":
                            response = await session.get(url)
                        elif method.upper() == "POST":
                            response = await session.post(url, json=payload)
                        elif method.upper() == "PUT":
                            response = await session.put(url, json=payload)
                        elif method.upper() == "DELETE":
                            response = await session.delete(url)
                        else:
                            raise ValueError(f"Unsupported HTTP method: {method}")

                        response.raise_for_status()
                        return response, server_info

                except (
                    aiohttp.ClientError,
                    aiohttp.ClientResponseError,
                    asyncio.TimeoutError,
                ) as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                    continue
        raise RuntimeError(
            f"Failed after {max_retries} retries each. " f"Last error: {last_exception}"
        )

    async def agenerate(self, req: LLMRequest) -> LLMResponse:
        raise NotImplementedError()

    async def aupdate_weights_from_disk(self, server_info: LLMServerInfo, path: str):
        raise NotImplementedError()

    async def ainit_weight_update_group(
        self, server_info: LLMServerInfo, group_meta: WeightUpdateGroupMeta
    ):
        raise NotImplementedError()

    async def aupdate_weights_from_distributed(
        self, server_info: LLMServerInfo, weight_meta: WeightMeta
    ):
        raise NotImplementedError()


@dataclass
class LLMClientFactory:
    """Factory class to create LLMClient instances."""

    args: TrainingArgs

    def make_client(self, config: LLMClientConfig) -> LLMClient:
        """Create an instance of LLMClient based on the specified type."""
        if self.args.rollout.server_backend == "sglang":
            from arealite.system.sglang_client import SGLangClient

            return SGLangClient(self.args, config)
        raise ValueError(f"Unknown LLMClient type: {self.args.rollout.server_backend}")
