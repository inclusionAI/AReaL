import abc
from dataclasses import dataclass

from arealite.api.cli_args import LLMClientConfig, TrainingArgs
from arealite.api.io_struct import (
    LLMRequest,
    LLMResponse,
    WeightMeta,
    WeightUpdateGroupMeta,
)


class LLMClient(abc.ABC):
    def __init__(self, args: TrainingArgs, client_config: LLMClientConfig):
        self.args = args
        self.client_config = client_config

    def generate(self, req: LLMRequest) -> LLMResponse:
        raise NotImplementedError()

    def update_weights_from_disk(self, path: str):
        raise NotImplementedError()

    def init_weight_update_group(self, group_meta: WeightUpdateGroupMeta):
        raise NotImplementedError()

    def update_weights_from_distributed(self, weight_meta: WeightMeta):
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
