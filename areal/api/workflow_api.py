from __future__ import annotations  # noqa

from typing import TYPE_CHECKING, Any

from areal.experimental.openai.types import InteractionWithTokenLogpReward

if TYPE_CHECKING:
    from areal.api.engine_api import InferenceEngine


class RolloutWorkflow:
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
