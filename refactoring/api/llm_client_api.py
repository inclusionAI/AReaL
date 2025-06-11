import abc
from dataclasses import dataclass

from refactoring.api.cli_args import LLMServiceConfig, TrainingArgs
from refactoring.api.engine_api import Engine
from refactoring.api.io_struct import LLMRequest, LLMResponse


class LLMClient(abc.ABC):
    def __init__(self, args: TrainingArgs, config: LLMServiceConfig):
        self.args = args
        self.config = config

    def generate(self, req: LLMRequest) -> LLMResponse:
        raise NotImplementedError()

    def update_weights_from(self, engine: Engine) -> None:
        """Update weights from the engine after an RL training step."""
        raise NotImplementedError()


@dataclass
class LLMClientFactory:
    """Factory class to create LLMClient instances."""

    args: TrainingArgs

    def make_client(self, config: LLMServiceConfig) -> LLMClient:
        """Create an instance of LLMClient based on the specified type."""
        if config.server_backend == "sglang":
            from xxx import SGLangLLMClient

            return SGLangLLMClient(self.args, config)
        else:
            raise ValueError(f"Unknown LLMClient type: {config.server_backend}")
