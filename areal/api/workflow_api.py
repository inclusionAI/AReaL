from __future__ import annotations  # noqa

from typing import TYPE_CHECKING, Any, Dict

from areal.experimental.openai.types import (
    CompletionWithTokenLogpReward,
    ResponseWithTokenLogpReward,
)

if TYPE_CHECKING:
    from areal.api.engine_api import InferenceEngine


class RolloutWorkflow:

    async def arun_episode(
        self, engine: "InferenceEngine", data: Dict[str, Any]
    ) -> (
        Dict[str, Any]
        | None
        | Dict[str, CompletionWithTokenLogpReward | ResponseWithTokenLogpReward]
    ):
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
        Dict[str, Any] | None | Dict[str, CompletionWithTokenLogpReward, ResponseWithTokenLogpReward]
            The trajectory result, None if rejected, or a dictionary of completion results
        """
        raise NotImplementedError()
