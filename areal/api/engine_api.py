from __future__ import annotations

import abc
from collections.abc import Callable
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any

from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api.io_struct import (
    LocalInfServerInfo,
    ModelRequest,
    ModelResponse,
    ParamSpec,
    WeightUpdateMeta,
)
from areal.api.train_engine import (
    TrainEngineComputeMixin,
    TrainEngineDistMixin,
    TrainEngineStateMixin,
)

if TYPE_CHECKING:
    from areal.api.workflow_api import RolloutWorkflow
    from areal.core.workflow_executor import WorkflowExecutor


class TrainEngine(
    TrainEngineDistMixin,
    TrainEngineStateMixin,
    TrainEngineComputeMixin,
    abc.ABC,
):
    pass


class InferenceEngine(abc.ABC):
    def initialize(self, *args, **kwargs):
        """Initialize environments and launch the background thread for asynchronous distributed inference.

        For remote inference engines, this serves as a client and connects to the inference servers.
        For local inference engines, this creates an LLM engine on the local GPU.

        Parameters
        ----------
        *args
            Variable length argument list
        **kwargs
            Arbitrary keyword arguments
        """
        raise NotImplementedError()

    def destroy(self):
        """Destroy the engine and release GPU memory for the local inference engine."""
        raise NotImplementedError()

    @property
    def workflow_executor(self) -> WorkflowExecutor:
        """Get the workflow executor of the inference engine."""
        raise NotImplementedError()

    def launch_server(self, server_args: dict[str, Any]) -> LocalInfServerInfo:
        """Launch a local inference server via subprocess and return its connection info.

        By default, an `InferenceEngine` instance acts as a client that connects to an existing
        remote inference server without occupying GPU resources. This is the typical usage in
        SPMD mode, where each training process has an attached inference client.

        This method enables launching a local inference server process, which is useful for:

        1. **Single-controller mode**: Launch a local server to serve the `InferenceEngine`
           instance with direct GPU worker control.

        2. **Standalone inference**: Use AReaL's inference engine in independent scripts or notebooks
           for running agentic workflows without managing separate server processes.

        Parameters
        ----------
        server_args : dict[str, Any]
            CLI arguments for the inference server (e.g., model path, GPU indices,
            port numbers, backend-specific settings)

        Returns
        -------
        LocalInfServerInfo
            Information about the launched server, including connection details and process metadata

        See Also
        --------
        teardown_server : Teardown the server launched by this method
        """
        raise NotImplementedError()

    def teardown_server(self):
        """Teardown the inference server launched by `launch_server`."""
        raise NotImplementedError()

    async def agenerate(self, req: ModelRequest) -> ModelResponse:
        """Asynchronously generate a response for the given request.

        Parameters
        ----------
        req : ModelRequest
            The model request containing input data and generation parameters

        Returns
        -------
        ModelResponse
            The generated response from the model
        """
        raise NotImplementedError()

    def init_weights_update_group(self, meta: WeightUpdateMeta) -> Future[None]:
        """Initialize the weight update process group for distributed weight updates.

        This method should be called before performing any weight updates to ensure
        that the necessary communication groups are set up correctly.

        Parameters
        ----------
        meta : WeightUpdateMeta
            Metadata containing information about the weight update, such as the
            type of communication backend and allocation mode.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by a subclass.

        Returns
        -------
        Future[None]
            A future object representing the asynchronous initialization operation.
        """
        raise NotImplementedError()

    def update_weights_from_distributed(
        self, meta: WeightUpdateMeta, param_specs: list[ParamSpec]
    ) -> Future[None]:
        """Update weights in the inference engine in a non-blocking manner.

        Parameters
        ----------
        meta : WeightUpdateMeta
            Metadata containing information about the weight update
        param_specs : List[ParamSpec]
            A list of parameter specifications for the weights to be updated

        Returns
        -------
        Future[None]
            A future object representing the asynchronous weight update operation
        """
        raise NotImplementedError()

    def update_weights_from_disk(self, meta: WeightUpdateMeta) -> Future[None]:
        """Update weights in the inference engine from disk in a non-blocking manner.

        Parameters
        ----------
        meta : WeightUpdateMeta
            Metadata containing information about the weight update

        Returns
        -------
        Future[None]
            A future object representing the asynchronous weight update operation
        """
        raise NotImplementedError()

    def set_version(self, version: int) -> None:
        """Set the current weight version in the inference engine.

        Parameters
        ----------
        version : int
            The weight version number to set
        """
        raise NotImplementedError()

    def get_version(self) -> int:
        """Get the current weight version in the inference engine.

        Returns
        -------
        int
            The current weight version number
        """
        raise NotImplementedError()

    def submit(
        self,
        data: dict[str, Any],
        workflow: RolloutWorkflow | type[RolloutWorkflow] | str,
        should_accept_fn: Callable | None = None,
        workflow_kwargs: dict[str, Any] | None = None,
    ) -> int:
        """Submit a request to the inference engine and return immediately.

        Should be used together with subsequent `wait`.

        Parameters
        ----------
        data : dict[str, Any]
            The input data for rollout. Used by the user's customized workflow implementation.
        workflow : RolloutWorkflow | type[RolloutWorkflow] | str
            The workflow to use for rollout generation. Can be:

            - An instance of RolloutWorkflow (for sharing resources between rollouts)
            - A RolloutWorkflow class type (will be instantiated with workflow_kwargs)
            - A string module path like "areal.workflow.rlvr.RLVRWorkflow" (will be imported
              and instantiated with workflow_kwargs)
        workflow_kwargs : dict[str, Any], optional
            Keyword arguments to pass to the workflow constructor when workflow is a type or string.
            Required when workflow is a type or string, ignored when workflow is an instance.
            By default None.
        should_accept_fn : Callable, optional
            A function used to decide whether to accept a specific trajectory, i.e., dynamic filtering.
            It takes a complete trajectory output by the workflow, and returns a bool, by default None.

        Returns
        -------
        int
            The id assigned to this task
        """
        raise NotImplementedError()

    def wait(
        self, count: int, timeout: float | None = None, raise_timeout: bool = True
    ) -> list[dict[str, Any] | None]:
        """Wait for a specified number of requests to complete, with a timeout.

        Should be used together with preceding `submit`.

        Parameters
        ----------
        count : int
            The number of accepted trajectories to wait for
        timeout : float, optional
            Timeout in seconds. Exceeding the timeout will raise a `TimeoutError`, by default None
        raise_timeout : bool, optional
            Whether to raise a `TimeoutError` when the timeout is exceeded,
            otherwise return an empty list, by default True

        Returns
        -------
        list[dict[str, Any] | None]
            A list of trajectory dictionaries. Each element may be None for rejected trajectories.

        Raises
        ------
        TimeoutError
            If the timeout is exceeded before enough trajectories are collected
        """
        raise NotImplementedError()

    def wait_for_task(
        self, task_id: int, timeout: float | None = None, raise_timeout: bool = True
    ) -> dict[str, Any] | None:
        """Wait for a specific task to complete by task_id.

        Parameters
        ----------
        task_id : int
            The task ID returned by submit()
        timeout : float | None, optional
            Timeout in seconds, by default None
        raise_timeout : bool, optional
            Whether to raise TimeoutError on timeout, by default True

        Returns
        -------
        dict[str, Any] | None
            Trajectory dict, or None if rejected or timeout with raise_timeout=False

        Raises
        ------
        ValueError
            If task_id was never submitted or already consumed
        TimeoutError
            If timeout expires and raise_timeout=True
        """
        raise NotImplementedError()

    def rollout_batch(
        self,
        data: list[dict[str, Any]],
        workflow: RolloutWorkflow | type[RolloutWorkflow] | str,
        workflow_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Submit a batch of requests to the inference engine and wait for the results.

        This method does not support asynchronous rollout and should be used for offline
        data collection or debugging, not in production experiments.

        See `workflow_api.py` for concrete implementation.

        Parameters
        ----------
        data : list[dict[str, Any]]
            A list of input data dictionaries for rollout
        workflow : RolloutWorkflow | type[RolloutWorkflow] | str
            The workflow to use for rollout generation. Can be:

            - An instance of RolloutWorkflow (for sharing resources between rollouts)
            - A RolloutWorkflow class type (will be instantiated with workflow_kwargs)
            - A string module path like "areal.workflow.rlvr.RLVRWorkflow" (will be imported
              and instantiated with workflow_kwargs)
        workflow_kwargs : dict[str, Any], optional
            Keyword arguments to pass to the workflow constructor when workflow is a type or string.
            Required when workflow is a type or string, ignored when workflow is an instance.
            By default None.

        Returns
        -------
        dict[str, Any]
            A concatenated batch of trajectory results
        """
        raise NotImplementedError()

    def prepare_batch(
        self,
        dataloader: StatefulDataLoader,
        workflow: RolloutWorkflow | type[RolloutWorkflow] | str,
        workflow_kwargs: dict[str, Any] | None = None,
        should_accept_fn: Callable | None = None,
    ) -> dict[str, Any]:
        """Asynchronously submit and wait until a full batch is ready with controlled staleness.

        See `workflow_api.py` for concrete implementation.

        .. warning::

            This method caches an internal data generator on the first call.
            The ``dataloader``, ``workflow``, ``workflow_kwargs``, and
            ``should_accept_fn`` parameters are captured at the first invocation
            and reused in all subsequent calls. Passing different arguments in
            later calls will **not** take effect.

            If you need to switch configurations mid-training, consider:

            - Using a separate inference engine instance
            - Using the :meth:`submit` / :meth:`wait` pattern for finer control

        Parameters
        ----------
        dataloader : StatefulDataLoader
            The data loader to pull data from for batch preparation
        workflow : RolloutWorkflow | type[RolloutWorkflow] | str
            The workflow to use for rollout generation. Can be:

            - An instance of RolloutWorkflow (for sharing resources between rollouts)
            - A RolloutWorkflow class type (will be instantiated with workflow_kwargs)
            - A string module path like "areal.workflow.rlvr.RLVRWorkflow" (will be imported
              and instantiated with workflow_kwargs)
        workflow_kwargs : dict[str, Any], optional
            Keyword arguments to pass to the workflow constructor when workflow is a type or string.
            Required when workflow is a type or string, ignored when workflow is an instance.
            By default None.
        should_accept_fn : Callable, optional
            A function to decide whether to accept a trajectory, by default None

        Returns
        -------
        dict[str, Any]
            A full batch of trajectory results with controlled staleness
        """
        raise NotImplementedError()

    def pause_generation(self):
        """Pause the generation of inference engine.

        Used during updating weights from distributed or disk.
        """
        raise NotImplementedError()

    def continue_generation(self):
        """Continue the generation of inference engine."""
        raise NotImplementedError()

    def pause(self):
        """Pause request submission for async rollout.

        Used during evaluation to prevent data over-generation.
        """
        raise NotImplementedError()

    def resume(self):
        """Resume request submission for async rollout."""
        raise NotImplementedError()

    def offload(self):
        """Offload model from GPU to CPU for inference engine."""
        raise NotImplementedError()

    def onload(self, tags: list[str] | None = None):
        """Onload model from CPU to GPU for inference engine.

        Parameters
        ----------
        tags : list[str], optional
            Tags to onload specific components. If None, onloads all components.
        """
        raise NotImplementedError()

    def export_stats(self) -> dict[str, float]:
        """Export the statistics recorded during workflow execution in the process.

        Workflow should only record scalar metrics like "rewards".
        These metrics will be reduced in the controller side.

        Note
        ----
        This method should be only called by the controller.

        Returns
        -------
        dict[str, float]
            The recorded scalar statistics.
        """
        raise NotImplementedError()
