# Copyright 2025 Ant Group Inc.
# Licensed under the Apache License, Version 2.0

import abc
import asyncio
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
            client_config.tokenizer_path
        )
        self._server_idx = 0

    def select_server(self):
        """Get an available healthy server."""
        servers = self.get_healthy_servers()
        assert self.client_config.schedule_policy == "round_robin"
        # Simple round-robin selection (could be improved with load balancing)
        server_info = servers[self._server_idx % len(servers)]
        self._server_idx += 1
        return server_info

    def get_healthy_servers(self):
        servers = self.registry.get_healthy_servers()
        if not servers:
            raise RuntimeError("No healthy SGLang servers available")
        return servers

    def request_with_retry(
        self,
        endpoint: str,
        payload: Optional[Dict[str, Any]] = None,
        method: str = "POST",
        max_retries: Optional[int] = None,
        timeout: Optional[float] = None,
        retry_delay: float = 1.0,
        target_server: Optional[LLMServerInfo] = None,
    ) -> tuple[requests.Response, LLMServerInfo]:
        """
        Send HTTP request to servers with retry logic and server switching.

        Args:
            endpoint: API endpoint (e.g., "/generate", "/health")
            payload: Request payload for POST/PUT requests
            method: HTTP method ("GET", "POST", "PUT", "DELETE")
            max_retries: Maximum number of retry attempts per server
            timeout: Request timeout in seconds
            retry_delay: Delay between retries in seconds

        Returns:
            tuple: (requests.Response, server_info) - Successful HTTP response and server info

        Raises:
            RuntimeError: If all servers fail after max retries
        """

        timeout = timeout or self.client_config.request_timeout
        last_exception = None
        max_retries = max_retries or self.client_config.request_retries

        # Try each server with retries
        for _ in range(max_retries):
            if target_server is None:
                server_info = self.select_server()
            else:
                server_info = target_server
            base_url = f"http://{server_info.host}:{server_info.port}"
            url = f"{base_url}{endpoint}"

            for attempt in range(max_retries):
                try:
                    if method.upper() == "GET":
                        response = requests.get(url, timeout=timeout)
                    elif method.upper() == "POST":
                        response = requests.post(url, json=payload, timeout=timeout)
                    elif method.upper() == "PUT":
                        response = requests.put(url, json=payload, timeout=timeout)
                    elif method.upper() == "DELETE":
                        response = requests.delete(url, timeout=timeout)
                    else:
                        raise ValueError(f"Unsupported HTTP method: {method}")

                    response.raise_for_status()
                    return response, server_info

                except (
                    requests.exceptions.RequestException,
                    requests.exceptions.HTTPError,
                ) as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                    continue

            # Mark server as potentially unhealthy after all retries failed
            # Note: Could implement more sophisticated health tracking here

        # All servers exhausted
        raise RuntimeError(
            f"All servers failed after {max_retries} retries each. "
            f"Last error: {last_exception}"
        )

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
        """
        Send async HTTP request to servers with retry logic and server switching.

        Args:
            endpoint: API endpoint (e.g., "/generate", "/health")
            payload: Request payload for POST/PUT requests
            method: HTTP method ("GET", "POST", "PUT", "DELETE")
            max_retries: Maximum number of retry attempts per server
            timeout: Request timeout in seconds
            retry_delay: Delay between retries in seconds

        Returns:
            tuple: (aiohttp.ClientResponse, server_info) - Successful HTTP response and server info

        Raises:
            RuntimeError: If all servers fail after max retries
        """

        timeout = timeout or self.client_config.request_timeout
        last_exception = None
        max_retries = max_retries or self.client_config.request_retries

        # Try each server with retries
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

            # Mark server as potentially unhealthy after all retries failed
            # Note: Could implement more sophisticated health tracking here

        # All servers exhausted
        raise RuntimeError(
            f"All servers failed after {max_retries} retries each. "
            f"Last error: {last_exception}"
        )

    def generate(self, req: LLMRequest) -> LLMResponse:
        raise NotImplementedError()

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
        if config.server_backend == "sglang":
            from arealite.impl.sglang_client import SGLangClient

            return SGLangClient(self.args, config)
        else:
            raise ValueError(f"Unknown LLMClient type: {config.server_backend}")
