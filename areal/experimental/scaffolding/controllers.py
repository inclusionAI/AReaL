"""
RLVR Controllers for Scaffolding Framework.

This module provides controllers for RLVR (Reinforcement Learning with Verifiable
Rewards) that integrate with TensorRT-LLM's scaffolding framework.

Key Components:
- RLVRRewardController: Controller that processes reward computation
- PipelineTrajectoryMaker: Controller that composes generation and reward pipelines
- ChatTracer: TaskCollection for tracing multi-turn chat conversations
- TraceTrajectoryMaker: Controller that traces ChatTask objects during rollout
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING, Any

from areal.api.reward_api import AsyncRewardWrapper
from areal.experimental.openai.cache import InteractionCache
from areal.experimental.openai.types import InteractionWithTokenLogpReward
from areal.experimental.scaffolding._compat import (
    ChatTask,
    Controller,
    GenerationTask,
    Task,
    TaskCollection,
    with_task_collection,
)
from areal.experimental.scaffolding.task import (
    ChatRewardTask,
    RLVRRewardTask,
    TraceGenerationTask,
)
from areal.utils import logging

if TYPE_CHECKING:
    pass

logger = logging.getLogger("RLVRControllers")


class RLVRRewardController(Controller):
    """Controller for computing RLVR (verifiable) rewards.

    This controller processes RLVRRewardTask objects and computes rewards
    using a provided reward function. The reward function should verify
    whether the generated answer is correct.

    The reward computation follows the pattern from RLVRWorkflow._compute_rewards:
    1. Decode output tokens to string (if needed)
    2. Call reward_fn(prompt_str, completion_str, input_tokens, output_tokens, **task_data)
    3. Store the reward in the task and update the interaction object

    Parameters
    ----------
    reward_fn : Callable
        The reward function that takes (prompt, completions, prompt_ids, completion_ids, **data)
        and returns a reward value (typically 0 or 1 for verifiable rewards).

    Example
    -------
    ```python
    from areal.reward.gsm8k import gsm8k_reward_fn

    reward_controller = RLVRRewardController(gsm8k_reward_fn)
    ```
    """

    class WorkerTag(Enum):
        """Worker tag for reward computation."""

        REWARD = "rlvr_reward"

    def __init__(self, reward_fn: Callable[..., Any]):
        """Initialize the RLVR reward controller.

        Parameters
        ----------
        reward_fn : Callable
            The reward function for verifying answers.
        """
        super().__init__()
        self.reward_fn = reward_fn
        self.async_reward_fn = AsyncRewardWrapper(reward_fn)
        self.scores: list[float] | None = None

    def process(self, tasks: list[Task], **kwargs) -> Any:
        """Process reward tasks and compute rewards.

        This method computes rewards for each task using the reward function.
        The rewards are stored in:
        1. task.reward - the computed reward value
        2. task.interaction.reward - if an interaction object is provided
        3. self.scores - list of all computed rewards

        Parameters
        ----------
        tasks : list[Task]
            List of RLVRRewardTask objects to process.
        **kwargs
            Additional keyword arguments.

        Yields
        ------
        list[Task]
            The processed tasks with rewards computed.
        """
        # Mark tasks with worker tag (for potential worker-based execution)
        for task in tasks:
            task.worker_tag = self.WorkerTag.REWARD

        # Compute rewards synchronously
        # Note: For async execution, this would be handled by a worker
        self.scores = []
        for task in tasks:
            if isinstance(task, RLVRRewardTask):
                reward = self._compute_reward(task)
                task.reward = reward
                self.scores.append(reward)

                # Update the interaction object if provided
                if task.interaction is not None:
                    task.interaction.reward = reward
            elif isinstance(task, GenerationTask):
                # For generation tasks, compute reward from customized fields
                reward = self._compute_reward_from_generation_task(task, **kwargs)
                task.customized_result_fields["reward"] = reward
                self.scores.append(reward)

        yield tasks

    def _compute_reward(self, task: RLVRRewardTask) -> float:
        """Compute reward for an RLVR reward task.

        Parameters
        ----------
        task : RLVRRewardTask
            The reward task containing prompt, completion, and task data.

        Returns
        -------
        float
            The computed reward value.
        """
        reward = self.reward_fn(
            task.prompt_str,
            task.completion_str,
            task.input_tokens,
            task.output_tokens,
            **task.task_data,
        )
        return float(reward)

    def _compute_reward_from_generation_task(
        self, task: GenerationTask, **kwargs
    ) -> float:
        """Compute reward from a generation task.

        Parameters
        ----------
        task : GenerationTask
            The completed generation task.
        **kwargs
            Should contain 'task_data' with ground truth.

        Returns
        -------
        float
            The computed reward value.
        """
        task_data = kwargs.get("task_data", {})
        prompt_str = kwargs.get("prompt_str", task.input_str or "")

        reward = self.reward_fn(
            prompt_str,
            task.output_str or "",
            list(task.input_tokens or []),
            list(task.output_tokens or []),
            **task_data,
        )
        return float(reward)

    async def aprocess(self, tasks: list[Task], **kwargs) -> Any:
        """Process reward tasks asynchronously.

        This method computes rewards asynchronously using AsyncRewardWrapper.

        Parameters
        ----------
        tasks : list[Task]
            List of RLVRRewardTask objects to process.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        list[Task]
            The processed tasks with rewards computed.
        """
        # Mark tasks with worker tag
        for task in tasks:
            task.worker_tag = self.WorkerTag.REWARD

        # Compute rewards asynchronously
        self.scores = []
        for task in tasks:
            if isinstance(task, RLVRRewardTask):
                reward = await self._acompute_reward(task)
                task.reward = reward
                self.scores.append(reward)

                # Update the interaction object if provided
                if task.interaction is not None:
                    task.interaction.reward = reward

        return tasks

    async def _acompute_reward(self, task: RLVRRewardTask) -> float:
        """Compute reward asynchronously for an RLVR reward task.

        Parameters
        ----------
        task : RLVRRewardTask
            The reward task containing prompt, completion, and task data.

        Returns
        -------
        float
            The computed reward value.
        """
        reward = await self.async_reward_fn(
            task.prompt_str,
            task.completion_str,
            task.input_tokens,
            task.output_tokens,
            **task.task_data,
        )
        return float(reward)


class ChatTracer(TaskCollection):
    """TaskCollection for tracing multi-turn chat conversations.

    This class traces ChatTask objects during the controller's process execution.
    A multi-turn conversation uses the same ChatTask object across multiple yields,
    allowing us to track the evolution of the conversation.

    The tracer:
    1. In `before_yield`: Records the state of ChatTask before worker execution
    2. In `after_yield`: Captures the new messages added by the worker and creates
       InteractionWithTokenLogpReward objects

    The traced results can be exported via `get_trace_results()`, which returns
    a dict[str, InteractionWithTokenLogpReward] similar to client.py's export_interactions.

    Parameters
    ----------
    reward_discount : float
        Discount factor for backward reward propagation across turns.
    export_style : str
        Export style for interactions: 'concat' (tree structure) or 'individual'.

    Example
    -------
    ```python
    tracer = ChatTracer(reward_discount=0.9, export_style="individual")
    # Used via with_task_collection decorator or TraceTrajectoryMaker
    ```
    """

    def __init__(
        self,
        reward_discount: float = 1.0,
        export_style: str = "individual",
    ):
        super().__init__()
        self.reward_discount = reward_discount
        self.export_style = export_style

        # Cache for storing interactions, similar to InteractionCache in client.py
        self._cache = InteractionCache()

    def before_yield(self, tasks: list[Task]):
        """Called before tasks are yielded to workers.

        Parameters
        ----------
        tasks : list[Task]
            List of tasks about to be yielded.
        """
        pass

    def after_yield(self, tasks: list[Task]):
        """Called after tasks return from workers.

        Creates InteractionWithTokenLogpReward objects for each ChatTask.
        Uses task.completion.id as the interaction ID.

        Parameters
        ----------
        tasks : list[Task]
            List of tasks that have been processed by workers.
        """
        for task in tasks:
            if not isinstance(task, ChatTask):
                continue

            interaction = self._create_interaction_from_chat_task(task)
            # Use completion.id as the interaction key
            completion_id = task.completion.id
            self._cache[completion_id] = interaction

    def _create_interaction_from_chat_task(
        self,
        task: ChatTask,
    ) -> InteractionWithTokenLogpReward:
        """Create an InteractionWithTokenLogpReward from a ChatTask.

        Parameters
        ----------
        task : ChatTask
            The ChatTask. Must contain a `completion` attribute
            with the ChatCompletion object.

        Returns
        -------
        InteractionWithTokenLogpReward
            The interaction object capturing this turn.
        """
        from areal.api.io_struct import ModelResponse

        # Extract all messages
        messages = [
            msg.to_dict() if hasattr(msg, "to_dict") else msg for msg in task.messages
        ]

        # Create ModelResponse from task data
        input_tokens = list(task.input_tokens or [])
        output_tokens = list(task.output_tokens or [])

        model_response = ModelResponse(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            output_logprobs=[0.0] * len(output_tokens),
            output_versions=[-1] * len(output_tokens),
        )

        # Get completion from task (ChatTask will contain the ChatCompletion)
        completion = task.completion

        interaction = InteractionWithTokenLogpReward(
            model_response=model_response,
            reward=None,
            messages=messages,
            output_message_list=[],
            completion=completion,
            chat_template_type=self.export_style,
        )

        return interaction

    def get_trace_results(self) -> dict[str, InteractionWithTokenLogpReward]:
        """Export traced interactions.

        Returns the traced interactions in the specified export style.
        Applies reward discount before export if configured.

        Returns
        -------
        dict[str, InteractionWithTokenLogpReward]
            Dictionary mapping interaction IDs to their data.

        See Also
        --------
        client.py : export_interactions method for similar functionality
        """
        if len(self._cache) == 0:
            return {}

        return self._cache.export_interactions(
            style=self.export_style,
            reward_discount=self.reward_discount,
        )

    def clear(self) -> None:
        """Clear all traced data."""
        self._cache.clear()


class PipelineTrajectoryMaker(Controller):
    """Controller that composes generation and reward controllers into a pipeline.

    This controller orchestrates the full RLVR pipeline:
    1. Run generation via the generation controller
    2. Compute rewards via the reward controller
    3. Assemble results into InteractionWithTokenLogpReward objects

    Parameters
    ----------
    generation_controller : Controller
        The controller for text generation (e.g., NativeGenerationController).
    reward_controller : RLVRRewardController
        The controller for reward computation.
    task_data : dict[str, Any]
        Task data containing ground truth (e.g., "answer" field) for reward computation.
    prompt_str : str
        The prompt string used for generation.

    Example
    -------
    ```python
    from tensorrt_llm.scaffolding import NativeGenerationController

    gen_controller = NativeGenerationController()
    reward_controller = RLVRRewardController(gsm8k_reward_fn)
    trajectory_maker = PipelineTrajectoryMaker(
        gen_controller,
        reward_controller,
        task_data={"answer": "42"},
        prompt_str="What is the answer?",
    )
    ```
    """

    def __init__(
        self,
        generation_controller: Controller,
        reward_controller: RLVRRewardController,
        task_data: dict[str, Any] | None = None,
        prompt_str: str = "",
    ):
        """Initialize the pipeline trajectory maker.

        Parameters
        ----------
        generation_controller : Controller
            The generation controller.
        reward_controller : RLVRRewardController
            The reward controller.
        task_data : dict[str, Any], optional
            Task data containing ground truth for reward computation.
        prompt_str : str, optional
            The prompt string used for generation.
        """
        super().__init__()
        self.generation_controller = generation_controller
        self.reward_controller = reward_controller
        self.task_data = task_data if task_data is not None else {}
        self.prompt_str = prompt_str

    def process(self, tasks: list[Task], **kwargs) -> Any:
        """Process tasks through the generation and reward pipeline.

        Parameters
        ----------
        tasks : list[Task]
            List of generation tasks to process.
        **kwargs
            Additional keyword arguments.

        Yields
        ------
        dict[str, InteractionWithTokenLogpReward]
            Dictionary mapping task IDs to their interaction results.
        """
        # Step 1: Run generation
        yield from self.generation_controller.process(tasks, **kwargs)

        reward_tasks = []
        interactions = {}

        for i, task in enumerate(tasks):
            if isinstance(task, GenerationTask):
                # Create interaction object
                interaction = self._create_interaction_from_task(task)
                task_id = f"task_{i}"
                interactions[task_id] = interaction

                # Create reward task using constructor-provided task_data and prompt_str
                reward_task = RLVRRewardTask.create_from_generation_task(
                    gen_task=task,
                    prompt_str=self.prompt_str or task.input_str or "",
                    task_data=self.task_data,
                    interaction=interaction,
                )
                reward_tasks.append(reward_task)

        # Step 3: Process reward tasks
        yield from self.reward_controller.process(reward_tasks, **kwargs)

        # The interactions now have rewards set
        # Return as the final result
        yield interactions

    def _create_interaction_from_task(
        self, task: GenerationTask
    ) -> InteractionWithTokenLogpReward:
        """Create an InteractionWithTokenLogpReward from a generation task.

        Parameters
        ----------
        task : GenerationTask
            The completed generation task.

        Returns
        -------
        InteractionWithTokenLogpReward
            The interaction object with model response data.
        """
        from areal.api.io_struct import ModelResponse

        # Build ModelResponse from task data
        input_tokens = list(task.input_tokens or [])
        output_tokens = list(task.output_tokens or [])
        output_logprobs = task.customized_result_fields.get("output_logprobs", [])
        output_versions = task.customized_result_fields.get("output_versions", [])

        # Create ModelResponse
        model_response = ModelResponse(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            output_logprobs=list(output_logprobs)
            if output_logprobs
            else [0.0] * len(output_tokens),
            output_versions=list(output_versions)
            if output_versions
            else [-1] * len(output_tokens),
        )

        # Create interaction
        interaction = InteractionWithTokenLogpReward(
            model_response=model_response,
            reward=None,  # Will be set by reward controller
        )

        return interaction


@with_task_collection("chat_tracer", ChatTracer)
class TraceTrajectoryMaker(Controller):
    """Controller that traces ChatTask objects during rollout using ChatTracer.

    This controller uses the @with_task_collection decorator to automatically
    apply ChatTracer's before_yield and after_yield hooks around each yield
    in the rollout controller's process execution.

    A multi-turn conversation uses the same ChatTask object, which is traced
    across all yields. The trace results are stored in the TraceGenerationTask
    after processing.

    Parameters
    ----------
    rollout_controller : Controller
        The controller for rollout (e.g., a chat or agent controller).
    reward_controller : Controller
        The controller for computing rewards on traced interactions.

    Example
    -------
    ```python
    from tensorrt_llm.scaffolding import NativeGenerationController

    chat_controller = SomeChatController()
    reward_controller = RLVRRewardController(gsm8k_reward_fn)

    trace_maker = TraceTrajectoryMaker(
        rollout_controller=chat_controller,
        reward_controller=reward_controller,
    )

    # Process tasks
    result = trace_maker.generate(prompt)

    # Or use process directly
    task = TraceGenerationTask.create_from_prompt(prompt)
    for _ in trace_maker.process([task]):
        pass
    trace_results = task.trace_results
    ```
    """

    def __init__(
        self,
        rollout_controller: Controller,
        reward_controller: Controller,
    ):
        """Initialize the trace trajectory maker.

        Parameters
        ----------
        rollout_controller : Controller
            The controller for rollout execution.
        reward_controller : Controller
            The controller for reward computation.
        """
        super().__init__()
        self.rollout_controller = rollout_controller
        self.reward_controller = reward_controller

    def process(self, tasks: list[Task], **kwargs) -> Any:
        """Process tasks through the rollout and reward pipeline with tracing.

        This method:
        1. Extracts the generation_task from the TraceGenerationTask
        2. Runs the rollout_controller.process() with ChatTracer tracing
        3. Gets trace results from the ChatTracer
        4. Creates ChatRewardTask objects for each traced interaction
        5. Runs the reward_controller.process() to compute rewards
        6. Stores the trace results in the original task

        Parameters
        ----------
        tasks : list[Task]
            List of TraceGenerationTask objects to process.
        **kwargs
            Additional keyword arguments.

        Yields
        ------
        Any
            Results from the controllers.
        """
        # Get the generation task from the first TraceGenerationTask
        task = tasks[0]
        if isinstance(task, TraceGenerationTask):
            generation_task = task.generation_task
        else:
            generation_task = task

        # Run rollout with tracing (ChatTracer hooks applied via decorator)
        yield from self.rollout_controller.process([generation_task], **kwargs)

        # Get trace results from the ChatTracer (registered via decorator)
        chat_tracer = self.task_collections["chat_tracer"]
        trace_results = chat_tracer.get_trace_results()

        # Create reward tasks for each traced interaction
        reward_tasks = [
            ChatRewardTask.create_from_trace_result(interaction_id, interaction)
            for interaction_id, interaction in trace_results.items()
        ]

        # Run reward computation
        if reward_tasks:
            yield from self.reward_controller.process(reward_tasks, **kwargs)

            # Update trace_results with computed rewards
            for reward_task in reward_tasks:
                if (
                    reward_task.interaction is not None
                    and reward_task.reward is not None
                ):
                    reward_task.interaction.reward = reward_task.reward

        # Store trace results in the original task
        if isinstance(task, TraceGenerationTask):
            task.trace_results = trace_results

    def generate(self, prompt: str, **kwargs) -> Any:
        """Generate with tracing from a prompt string.

        Parameters
        ----------
        prompt : str
            The input prompt.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        Any
            The scaffolding output.
        """
        task = TraceGenerationTask.create_from_prompt(prompt)

        yield from self.process([task], **kwargs)

        return task.create_scaffolding_output()
