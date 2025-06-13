import abc
from dataclasses import dataclass
from typing import Any, Optional

from gymnasium.core import ObsType

from arealite.api.cli_args import AgentConfig, LLMClientConfig, TrainingArgs
from arealite.api.io_struct import AgentInferOutput
from arealite.api.llm_client_api import LLMClient, LLMClientFactory


class Agent(abc.ABC):
    def __init__(
        self,
        args: TrainingArgs,
        client_config: LLMClientConfig,
        agent_config: AgentConfig,
    ):
        self.args = args
        self.agent_config = agent_config

        # Create an LLM client to generate actions
        client_factory = LLMClientFactory(args)
        self.llm_client: LLMClient = client_factory.make_client(client_config)

    def act(self, obs: ObsType) -> AgentInferOutput:
        """Given an observation, return an action and data used for RL training."""
        raise NotImplementedError()

    def reset(self) -> None:
        """Resets the agent's memory."""
        raise NotImplementedError()


@dataclass
class AgentFactory:
    args: TrainingArgs
    client_config: LLMClientConfig

    def make_agent(self, config: AgentConfig) -> Agent:
        if config.type == "math-code-single-step":
            from arealite.impl.agent.math_code_single_step_agent import (
                MathCodeSingleStepAgent,
            )

            return MathCodeSingleStepAgent(
                self.args,
                self.client_config,
                config,
            )
        else:
            raise NotImplementedError(f"Unknown agent type: {config.type}")
