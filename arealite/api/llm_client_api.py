import abc
import asyncio
from dataclasses import dataclass

import transformers

from arealite.api.cli_args import LLMClientConfig, TrainingArgs
from arealite.api.io_struct import (
    LLMRequest,
    LLMResponse,
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
        servers = self.registry.get_healthy_servers()
        if not servers:
            raise RuntimeError("No healthy SGLang servers available")

        assert self.client_config.schedule_policy == "round_robin"
        # Simple round-robin selection (could be improved with load balancing)
        server_info = servers[self._server_idx % len(servers)]
        self._server_idx += 1
        return server_info

    def generate(self, req: LLMRequest) -> LLMResponse:
        raise NotImplementedError()

    async def agenerate(self, req: LLMRequest) -> LLMResponse:
        """A trick to make an async generation function."""
        return await asyncio.to_thread(self.generate, req)

    async def aupdate_weights_from_disk(self, path: str):
        raise NotImplementedError()

    async def ainit_weight_update_group(self, group_meta: WeightUpdateGroupMeta):
        raise NotImplementedError()

    async def aupdate_weights_from_distributed(self, weight_meta: WeightMeta):
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
