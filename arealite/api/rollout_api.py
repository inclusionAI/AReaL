import abc
import asyncio
from dataclasses import dataclass
from typing import Any, Optional, SupportsFloat

from gymnasium import Env
from gymnasium.core import ActType, ObsType
from gymnasium.utils import seeding

from arealite.api.cli_args import (
    GenerationHyperparameters,
    LLMClientConfig,
    RolloutWorkflowConfig,
    TrainingArgs,
)
from arealite.api.io_struct import AgentInferInput, AgentInferOutput, Trajectory
from arealite.api.llm_client_api import LLMClient, LLMClientFactory


class Agent(abc.ABC):
    def __init__(
        self,
        args: TrainingArgs,
        llm_client: LLMClient | None = None,
    ):
        self.args = args
        self.llm_client = llm_client

    def act(self, inp: AgentInferInput) -> AgentInferOutput:
        """Given an observation, return an action and data used for RL training."""
        raise NotImplementedError()

    def reset(self) -> None:
        """Resets the agent's memory."""
        raise NotImplementedError()


# Re-export the gymnasium environment class
class Environment(abc.ABC, Env):
    def __init__(self, args: TrainingArgs):
        self.args = args

    @abc.abstractmethod
    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        raise NotImplementedError()

    @abc.abstractmethod
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:  # type: ignore
        # Initialize the RNG if the seed is manually passed
        if seed is not None:
            self._np_random, self._np_random_seed = seeding.np_random(seed)


class RolloutWorkflow(abc.ABC):

    def __init__(
        self,
        args: TrainingArgs,
        config: RolloutWorkflowConfig,
        agent: Agent | None = None,
        env: Environment | None = None,
    ):
        self.args = args
        self.config = config

        self.agent = agent
        self.env = env

    def run_episode(
        self,
        gconfig: GenerationHyperparameters,
        env_option: Optional[Any] = None,
        seed: Optional[int] = None,
    ) -> Trajectory:
        """Run a single episode and return the trajectory."""
        raise NotImplementedError()

    async def run_episode_async(
        self,
        gconfig: GenerationHyperparameters,
        env_option: Optional[Any] = None,
        seed: Optional[int] = None,
    ) -> Trajectory:
        """Run a single episode asynchronously and return trajectory."""
        # A trick to convert a sync function to async function
        return await asyncio.to_thread(self.run_episode(gconfig, env_option, seed))


@dataclass
class RolloutWorkflowFactory:
    args: TrainingArgs

    def make_workflow(self, config: RolloutWorkflowConfig) -> RolloutWorkflow:
        client = LLMClientFactory(self.args).make_client(self.args.rollout.llm_client)
        if config.type == "rlvr":
            from arealite.impl.rlvr.rlvr_workflow import RlvrWorkflow

            return RlvrWorkflow(
                self.args,
                config=config,
                llm_client=client,
            )
        if config.type == "math_code_single_step":
            from arealite.impl.agentic.math_code_single_step import (
                MathCodeAgent,
                MathCodeSingleStepEnv,
                MathCodeSingleStepWorkflow,
            )

            agent = MathCodeAgent(self.args, llm_client=client)
            env = MathCodeSingleStepEnv(
                self.args,
                solution_path=config.math_code_single_step.solution_path,
            )

            return MathCodeSingleStepWorkflow(
                self.args,
                config=config,
                agent=agent,
                env=env,
            )
        raise NotImplementedError(f"Unknown agent type: {config.type}")
