from typing import TYPE_CHECKING, Any, Dict

from tensordict import TensorDict

if TYPE_CHECKING:
    from arealite.api.engine_api import InferenceEngine


class RolloutWorkflow:

    async def arun_episode(
        self, engine: "InferenceEngine", data: Dict[str, Any]
    ) -> TensorDict:
        """Run a single episode of the workflow.

        See concrete example implementations under the `arealite/workflow` directory.
        """
        raise NotImplementedError()
