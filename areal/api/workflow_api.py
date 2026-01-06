from __future__ import annotations  # noqa

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Union


if TYPE_CHECKING:
    from areal.api.engine_api import InferenceEngine
    from areal.experimental.openai.types import InteractionWithTokenLogpReward


class RolloutWorkflow(ABC):
    @abstractmethod
    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, Any] | None | dict[str, InteractionWithTokenLogpReward]:
        """Run a single episode of the workflow.

        Note
        ----
        Returning `None` implies that this trajectory is rejected and will not be used for training.

        See concrete example implementations under the `areal/workflow` directory.

        Parameters
        ----------
        engine : InferenceEngine
            The inference engine to use for generating responses
        data : Dict[str, Any]
            Input data for the workflow episode

        Returns
        -------
        Dict[str, Any] | None | Dict[str, InteractionWithTokenLogpReward]
            The trajectory result, None if rejected, or a dictionary of completion results
        """
        raise NotImplementedError()


class AgentWorkflow(ABC):
    async def run(
        self, base_url: str, data: dict[str, Any]
    ) -> dict[str, float] | float:
        """Run an agent with any SDK, e.g., OpenAI SDK.

        Example:
        ```python
        class MyOpenAIAgent(AgentWorkflow):
            async def run(self, base_url, data):
                from openai import AsyncOpenAI
                async with AsyncOpenAI(base_url=base_url) as client:
                    resp = await client.responses.create(input=data["input"])
                reward = compute_reward(resp, data)
                return reward
        ```

        Parameters
        ----------
        base_url : str
            The base URL of the OpenAI-compatible proxy server
        data : dict[str, Any]
            Input data for the agent workflow

        Returns
        -------
        dict[str, float] | float
            The final reward or a dictionary of reward keyed by response ID
        """
        raise NotImplementedError()


# Type alias for workflow parameter across the stack.
# Accepts RolloutWorkflow instances/classes, string import paths, or AgentWorkflow instances/classes.
WorkflowLike = Union[
    "RolloutWorkflow",
    type["RolloutWorkflow"],
    str,
    "AgentWorkflow",
    type["AgentWorkflow"],
]
