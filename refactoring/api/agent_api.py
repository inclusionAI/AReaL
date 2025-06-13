import abc
from dataclasses import dataclass
from typing import Any, Optional

from gymnasium.core import ActType, ObsType

from refactoring.api.cli_args import AgentConfig, TrainingArgs
from refactoring.api.llm_client_api import LLMClient, LLMClientFactory


class Agent(abc.ABC):
    def __init__(self, args: TrainingArgs, config: AgentConfig):
        self.args = args
        self.config = config
        if args.inf_service is not None:
            client_factory = LLMClientFactory(args)
            self.llm_client: Optional[LLMClient] = client_factory.make_client(
                args.inf_service
            )
        else:
            self.llm_client: Optional[LLMClient] = None

    def act(self, obs: ObsType) -> ActType:
        """Given an observation, return an action."""
        raise NotImplementedError()

    def reset(self):
        """Resets the agent's memory."""
        raise NotImplementedError()


@dataclass
class AgentFactory:
    args: TrainingArgs

    def make_agent(self, config: AgentConfig) -> Agent:
        if config.type == "math-code-single-step":
            from refactoring.impl.agent.math_code_single_step_agent import (
                MathCodeSingleStepAgent,
            )

            return MathCodeSingleStepAgent(self.args, config)
        else:
            raise NotImplementedError(f"Unknown agent type: {config.type}")
