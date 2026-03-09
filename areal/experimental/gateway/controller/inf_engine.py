"""GatewayInfEngine — InferenceEngine implementation that routes through the gateway HTTP stack."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from threading import Lock
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from areal.api.io_struct import ModelRequest, ModelResponse
    from areal.experimental.gateway.controller.config import GatewayControllerConfig


class _GatewayInfEngineConfig:
    """Adapter that exposes GatewayControllerConfig fields under the names
    expected by WorkflowExecutor (which reads InferenceEngineConfig attributes).

    This avoids copying the real InferenceEngineConfig dataclass and keeps
    GatewayControllerConfig as the single source of truth.
    """

    def __init__(self, cfg: GatewayControllerConfig) -> None:
        self._cfg = cfg

    # Fields read by WorkflowExecutor ------------------------------------------
    @property
    def consumer_batch_size(self) -> int:
        return self._cfg.consumer_batch_size

    @property
    def max_concurrent_rollouts(self) -> int | None:
        return self._cfg.max_concurrent_rollouts

    @property
    def max_head_offpolicyness(self) -> int:
        return self._cfg.max_head_offpolicyness

    @property
    def queue_size(self) -> int | None:
        return self._cfg.queue_size

    @property
    def enable_rollout_tracing(self) -> bool:
        return self._cfg.enable_rollout_tracing

    @property
    def check_trajectory_format(self) -> bool:
        return self._cfg.check_trajectory_format

    @property
    def tokenizer_path(self) -> str:
        return self._cfg.tokenizer_path

    @property
    def fileroot(self) -> str | None:
        return self._cfg.fileroot

    @property
    def experiment_name(self) -> str | None:
        return self._cfg.experiment_name

    @property
    def trial_name(self) -> str | None:
        return self._cfg.trial_name

    @property
    def request_timeout(self) -> float:
        return self._cfg.request_timeout

    @property
    def pause_grace_period(self) -> float:
        return self._cfg.pause_grace_period


class GatewayInfEngine:
    """Inference engine that routes all calls through the gateway HTTP stack.

    This class is duck-type compatible with ``RemoteInfEngine`` to the extent
    needed by ``WorkflowExecutor`` — it provides ``get_version()``,
    ``set_version()``, and ``agenerate()``.  The ``WorkflowExecutor`` passes
    this engine to ``workflow.arun_episode(engine, data)`` which in turn calls
    ``engine.agenerate(req)`` for each generation.
    """

    def __init__(
        self,
        gateway_addr: str,
        config: GatewayControllerConfig,
    ) -> None:
        self.gateway_addr = gateway_addr  # e.g. "http://127.0.0.1:8080"
        self.config = config
        # Adapter config for WorkflowExecutor compatibility
        self.wf_config = _GatewayInfEngineConfig(config)

        self._version = 0
        self._version_lock = Lock()

        self._workflow_executor = None
        self._initialized = False

    @property
    def initialized(self) -> bool:
        return self._initialized

    # -- Version management ------------------------------------------------

    def set_version(self, version: int) -> None:
        with self._version_lock:
            self._version = version

    def get_version(self) -> int:
        with self._version_lock:
            return self._version

    # -- Core generation ---------------------------------------------------

    async def agenerate(self, req: ModelRequest) -> ModelResponse:
        """Send a generation request through the gateway HTTP stack.

        The gateway routes the request to a data proxy (via the router),
        which forwards it to a co-located SGLang server.  The response is
        streamed back as SSE chunks which we accumulate into a
        ``ModelResponse``.
        """
        from areal.api.io_struct import ModelResponse

        start_time = time.perf_counter()

        # Build payload matching the gateway /generate endpoint format
        payload: dict[str, Any] = {
            "input_ids": req.input_ids,
            "sampling_params": {
                "max_new_tokens": req.gconfig.max_new_tokens,
                "temperature": req.gconfig.temperature,
                "top_p": req.gconfig.top_p,
                "skip_special_tokens": False,
            },
        }
        if hasattr(req.gconfig, "stop_token_ids") and req.gconfig.stop_token_ids:
            payload["sampling_params"]["stop_token_ids"] = req.gconfig.stop_token_ids

        import aiohttp

        accumulated_tokens: list[int] = []
        accumulated_logprobs: list[float] = []
        stop_reason: str | None = None

        url = f"{self.gateway_addr}/generate"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.admin_api_key}",
        }

        timeout = aiohttp.ClientTimeout(total=self.config.request_timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload, headers=headers) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(
                        f"Gateway /generate returned {resp.status}: {text}"
                    )

                import json as json_mod

                # Parse SSE stream
                buffer = b""
                async for chunk in resp.content.iter_any():
                    buffer += chunk
                    while b"\n\n" in buffer:
                        frame, buffer = buffer.split(b"\n\n", 1)
                        for line in frame.split(b"\n"):
                            line = line.strip()
                            if line.startswith(b"data: "):
                                data_str = line[6:].decode()
                                if data_str == "[DONE]":
                                    break
                                try:
                                    data = json_mod.loads(data_str)
                                except json_mod.JSONDecodeError:
                                    continue
                                token_id = data.get("token")
                                logprob = data.get("logprob", 0.0)
                                if token_id is not None:
                                    accumulated_tokens.append(token_id)
                                    accumulated_logprobs.append(logprob)
                                if data.get("finished"):
                                    stop_reason = data.get("stop_reason", "stop")

        if stop_reason is None:
            stop_reason = "stop"

        latency = time.perf_counter() - start_time

        response = ModelResponse(
            input_tokens=req.input_ids,
            output_tokens=accumulated_tokens,
            output_logprobs=accumulated_logprobs,
            output_versions=[self.get_version()] * len(accumulated_tokens),
            stop_reason=stop_reason,
            latency=latency,
            ttft=latency,
            tokenizer=getattr(req, "tokenizer", None),
            processor=getattr(req, "processor", None),
        )
        return response

    # -- WorkflowExecutor lifecycle ----------------------------------------

    def initialize(self, train_data_parallel_size: int | None = None) -> None:
        """Create the WorkflowExecutor (mirrors RemoteInfEngine.initialize)."""
        from areal.infra.workflow_executor import WorkflowExecutor

        self._workflow_executor = WorkflowExecutor(
            config=self.wf_config,
            inference_engine=self,
        )
        self._workflow_executor.initialize(
            train_data_parallel_size=train_data_parallel_size,
        )
        self._initialized = True

    def destroy(self) -> None:
        if self._workflow_executor is not None:
            self._workflow_executor.destroy()
        self._initialized = False

    @property
    def workflow_executor(self):
        if self._workflow_executor is None:
            raise RuntimeError("GatewayInfEngine not initialized")
        return self._workflow_executor

    # -- Workflow resolution helpers ------------------------------------

    def _wrap_openai_agent(self, agent: Any, proxy_addr: str):
        """Wrap an agent workflow in OpenAIProxyWorkflow (HTTP mode only).

        Parameters
        ----------
        agent : Any | None
            The agent workflow to wrap (any class with async run() method).
            ``None`` is valid when ``mode='online'``.
        proxy_addr : str
            HTTP address of the proxy server (required)
        """
        from areal.experimental.openai import OpenAIProxyWorkflow

        openai_cfg = self.config.openai
        # Use config attributes if provided, otherwise fall back to
        # OpenAIProxyConfig defaults (avoids importing areal.api.cli_args
        # which triggers PEP 695 syntax errors on Python < 3.12).
        mode = getattr(openai_cfg, "mode", "inline")
        admin_api_key = getattr(openai_cfg, "admin_api_key", "areal-admin-key")
        turn_discount = getattr(openai_cfg, "turn_discount", 1.0)
        export_style = getattr(openai_cfg, "export_style", "individual")
        subproc_max_workers = getattr(openai_cfg, "subproc_max_workers", 4)

        return OpenAIProxyWorkflow(
            mode=mode,
            agent=agent,
            proxy_addr=proxy_addr,
            admin_api_key=admin_api_key,
            discount=turn_discount,
            export_style=export_style,
            subproc_max_workers=subproc_max_workers,
            proxy_gateway_addr=self.gateway_addr,
        )

    @staticmethod
    def _resolve_workflow(
        workflow,
        workflow_kwargs=None,
        group_size=1,
        proxy_addr=None,
        engine=None,
    ):
        """Resolve a WorkflowLike to a RolloutWorkflow instance.

        Handles both RolloutWorkflow types (cases 1-3) and agent-like
        workflows that need wrapping in OpenAIProxyWorkflow (cases 4-5).

        Parameters
        ----------
        workflow : WorkflowLike
            A RolloutWorkflow instance, class, import path string,
            agent class, or agent instance.
        workflow_kwargs : dict, optional
            Keyword arguments passed to the workflow/agent constructor.
        group_size : int
            Number of times to run the workflow per input.
        proxy_addr : str, optional
            HTTP address of the proxy server, required for agent workflows.
        engine : GatewayInfEngine, optional
            The engine instance, required for agent workflows (_wrap_openai_agent).
        """
        from areal.api.workflow_api import RolloutWorkflow
        from areal.utils.dynamic_import import import_from_string

        if workflow is None:
            raise ValueError("workflow must be specified")

        resolved: RolloutWorkflow

        # 1. Already a RolloutWorkflow instance
        if isinstance(workflow, RolloutWorkflow):
            resolved = workflow

        # 2. RolloutWorkflow class
        elif isinstance(workflow, type) and issubclass(workflow, RolloutWorkflow):
            if workflow_kwargs is None:
                raise ValueError("workflow_kwargs required when workflow is a class")
            resolved = workflow(**workflow_kwargs)

        # 3. String import path
        elif isinstance(workflow, str):
            imported = import_from_string(workflow)
            if isinstance(imported, type) and issubclass(imported, RolloutWorkflow):
                if workflow_kwargs is None:
                    raise ValueError("workflow_kwargs required when workflow is a class")
                resolved = imported(**workflow_kwargs)
            elif isinstance(imported, RolloutWorkflow):
                resolved = imported
            else:
                # Treat as agent-like workflow (needs proxy wrapping)
                if proxy_addr is None or engine is None:
                    raise ValueError(
                        f"proxy_addr and engine are required for agent workflows "
                        f"(non-RolloutWorkflow). Got workflow={workflow!r}"
                    )
                if isinstance(imported, type):
                    agent = imported(**(workflow_kwargs or {}))
                else:
                    agent = imported
                resolved = engine._wrap_openai_agent(agent, proxy_addr=proxy_addr)

        # 4. Callable class (agent-like workflow)
        elif isinstance(workflow, type):
            if proxy_addr is None or engine is None:
                raise ValueError(
                    "proxy_addr and engine are required for agent workflows "
                    "(non-RolloutWorkflow). "
                    "Ensure proxy workers are initialized via RolloutController.start_proxy()."
                )
            agent = workflow(**(workflow_kwargs or {}))
            resolved = engine._wrap_openai_agent(agent, proxy_addr=proxy_addr)

        # 5. Instance of agent-like workflow
        else:
            if proxy_addr is None or engine is None:
                raise ValueError(
                    "proxy_addr and engine are required for agent workflows "
                    "(non-RolloutWorkflow). "
                    "Ensure proxy workers are initialized via RolloutController.start_proxy()."
                )
            resolved = engine._wrap_openai_agent(workflow, proxy_addr=proxy_addr)

        if group_size > 1:
            from areal.infra.remote_inf_engine import GroupedRolloutWorkflow
            import logging as _logging

            resolved = GroupedRolloutWorkflow(
                resolved, group_size, _logging.getLogger("GatewayInfEngine")
            )

        return resolved

    @staticmethod
    def _resolve_should_accept_fn(should_accept_fn):
        """Resolve should_accept_fn to a callable or None."""
        if should_accept_fn is None or callable(should_accept_fn):
            return should_accept_fn
        if isinstance(should_accept_fn, str):
            from areal.utils.dynamic_import import import_from_string

            func = import_from_string(should_accept_fn)
            if not callable(func):
                raise TypeError(f"Imported {should_accept_fn!r} is not callable")
            return func
        raise TypeError(f"Invalid should_accept_fn type: {type(should_accept_fn)}")

    # -- Delegations to WorkflowExecutor -----------------------------------

    def submit(
        self,
        data: dict[str, Any],
        workflow,
        workflow_kwargs: dict[str, Any] | None = None,
        should_accept_fn: Callable[[dict[str, Any]], bool] | str | None = None,
        group_size: int = 1,
        task_id: int | None = None,
        is_eval: bool = False,
        proxy_addr: str | None = None,
        **kwargs: Any,
    ) -> int:
        if proxy_addr is None:
            proxy_addr = self.gateway_addr
        resolved_workflow = self._resolve_workflow(
            workflow, workflow_kwargs, group_size,
            proxy_addr=proxy_addr, engine=self,
        )
        resolved_accept_fn = self._resolve_should_accept_fn(should_accept_fn)
        return self.workflow_executor.submit(
            data,
            workflow=resolved_workflow,
            should_accept_fn=resolved_accept_fn,
            task_id=task_id,
            is_eval=is_eval,
        )

    def wait(
        self,
        count: int,
        timeout: float | None = None,
        raise_timeout: bool = True,
    ) -> list[dict[str, Any] | None]:
        return self.workflow_executor.wait(
            count, timeout=timeout, raise_timeout=raise_timeout
        )

    def wait_for_task(
        self,
        task_id: int,
        timeout: float | None = None,
        raise_timeout: bool = True,
    ) -> dict[str, Any] | None:
        return self.workflow_executor.wait_for_task(task_id, timeout, raise_timeout)

    def rollout_batch(
        self,
        data: list[dict[str, Any]],
        workflow,
        workflow_kwargs: dict[str, Any] | None = None,
        group_size: int = 1,
        proxy_addr: str | None = None,
    ) -> list[dict[str, Any]]:
        if proxy_addr is None:
            proxy_addr = self.gateway_addr
        resolved_workflow = self._resolve_workflow(
            workflow, workflow_kwargs, group_size,
            proxy_addr=proxy_addr, engine=self,
        )
        return self.workflow_executor.rollout_batch(
            data=data, workflow=resolved_workflow
        )

    def prepare_batch(
        self,
        dataloader,
        workflow,
        workflow_kwargs: dict[str, Any] | None = None,
        should_accept_fn: Callable[[dict[str, Any]], bool] | str | None = None,
        group_size: int = 1,
        dynamic_bs: bool = False,
        proxy_addr: str | None = None,
    ) -> list[dict[str, Any]]:
        if proxy_addr is None:
            proxy_addr = self.gateway_addr
        resolved_workflow = self._resolve_workflow(
            workflow, workflow_kwargs, group_size,
            proxy_addr=proxy_addr, engine=self,
        )
        resolved_accept_fn = self._resolve_should_accept_fn(should_accept_fn)
        return self.workflow_executor.prepare_batch(
            dataloader=dataloader,
            workflow=resolved_workflow,
            should_accept_fn=resolved_accept_fn,
            dynamic_bs=dynamic_bs,
        )

    def pause(self) -> None:
        """Pause the workflow executor (prevents new task dispatching)."""
        return self.workflow_executor.pause()

    def resume(self) -> None:
        """Resume the workflow executor."""
        return self.workflow_executor.resume()
