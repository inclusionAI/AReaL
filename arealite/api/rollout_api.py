# Copyright 2025 Ant Group Inc.
# Licensed under the Apache License, Version 2.0

import abc
from dataclasses import dataclass
from typing import Any, Callable, Optional, SupportsFloat

from gymnasium import Env
from gymnasium.core import ActType, ObsType
from gymnasium.utils import seeding

from arealite.api.cli_args import (
    GenerationHyperparameters,
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

    async def aact(self, inp: AgentInferInput) -> AgentInferOutput:
        """Async version of act. Given an observation, return an action and data used for RL training."""
        raise NotImplementedError()

    def reset(self) -> None:
        """Resets the agent's memory."""
        raise NotImplementedError()

    async def areset(self) -> None:
        """Async version of reset. Resets the agent's memory."""
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
        reward_func: Callable | None = None,
    ):
        self.args = args
        self.config = config

        # Used in agentic scenarios
        self.agent = agent
        self.env = env

        # Used in RLVR
        self.reward_func = reward_func

    def run_episode(
        self,
        gconfig: GenerationHyperparameters,
        env_option: Optional[Any] = None,
        seed: Optional[int] = None,
    ) -> Trajectory:
        """Run a single episode and return the trajectory."""
        raise NotImplementedError()

    async def arun_episode(
        self,
        gconfig: GenerationHyperparameters,
        env_option: Optional[Any] = None,
        seed: Optional[int] = None,
    ) -> Trajectory:
        """Async version of run_episode. Run a single episode and return the trajectory."""
        raise NotImplementedError()


@dataclass
class RolloutWorkflowFactory:
    args: TrainingArgs

    def make_workflow(self, config: RolloutWorkflowConfig) -> RolloutWorkflow:
        client = LLMClientFactory(self.args).make_client(self.args.rollout.llm_client)
        if config.type == "rlvr":
            from arealite.impl.rlvr.rlvr_workflow import RlvrWorkflow

            rlvr_config = config.rlvr
            if rlvr_config.reward_type == "math":
                from arealite.impl.rlvr.math_reward import get_math_reward_fn

                reward_fn = get_math_reward_fn(rlvr_config.solution_path)
            elif rlvr_config.reward_type == "code":
                from arealite.impl.rlvr.code_reward import get_code_reward_fn

                reward_fn = get_code_reward_fn(rlvr_config.solution_path)
            else:
                raise NotImplementedError(
                    f"Unknown reward type: {rlvr_config.reward_type}"
                )

            return RlvrWorkflow(
                self.args,
                config=config,
                llm_client=client,
                reward_fn=reward_fn,
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
