import abc
from dataclasses import dataclass

from arealite.api.cli_args import LLMClientConfig, TrainingArgs
from arealite.api.engine_api import SPMDWrapper
from arealite.api.io_struct import LLMRequest, LLMResponse


class LLMClient(abc.ABC):
    def __init__(self, args: TrainingArgs):
        self.args = args

    def generate(self, req: LLMRequest) -> LLMResponse:
        raise NotImplementedError()

    def update_weights_from(self, engine: SPMDWrapper) -> None:
        """Update weights from the engine after an RL training step."""
        raise NotImplementedError()


@dataclass
class LLMClientFactory:
    """Factory class to create LLMClient instances."""

    args: TrainingArgs

    def make_client(self, config: LLMClientConfig) -> LLMClient:
        """Create an instance of LLMClient based on the specified type."""
        if config.server_backend == "sglang":
            from xxx import SGLangLLMClient

            return SGLangLLMClient(self.args)
        else:
            raise ValueError(f"Unknown LLMClient type: {config.server_backend}")
