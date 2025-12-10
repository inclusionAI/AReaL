"""
NOTE: the proxy server is under development and in experimental stage, the interface are subject to change.
"""

import asyncio
import inspect
import logging
import threading
import time
from collections.abc import Awaitable, Callable
from copy import deepcopy
from dataclasses import dataclass, field
from types import TracebackType
from typing import TYPE_CHECKING, Any, TypedDict

import aiohttp
import requests
import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel

from openai.types.chat import ChatCompletion, CompletionCreateParams
from openai.types.responses import Response, ResponseCreateParams

from areal.experimental.openai.cache import InteractionCache
from areal.experimental.openai.client import ArealOpenAI
from areal.utils.logging import getLogger
from areal.utils.network import find_free_ports, gethostip
from areal.utils.proxy_utils import (
    CHAT_COMPLETIONS_PATHNAME,
    RESPONSES_PATHNAME,
    RL_END_SESSION_PATHNAME,
    RL_FINISH_TASK_PATHNAME,
    RL_SET_REWARD_PATHNAME,
    RL_START_SESSION_PATHNAME,
    AReaLFinishTaskRequest,
    AReaLSetRewardRequest,
    AReaLStartSessionRequest,
    ensure_end_with_slash,
    get_retry_strategy,
    post_json,
    post_json_with_retry,
    set_interaction_reward,
    set_last_interaction_reward,
    validate_json_request,
)

if TYPE_CHECKING:
    from areal.experimental.openai.types import InteractionWithTokenLogpReward


@get_retry_strategy(allowed_attempt=-1)
async def _start_session(*args, **kwargs):
    return await post_json(*args, **kwargs)


class SessionData:
    def __init__(
        self,
        id: str,
        completed: bool = False,
        completions: InteractionCache | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        messages: list[dict] | None = None,
    ):
        self.id = id
        self.completed = completed
        self.completions = completions or InteractionCache()
        self.start_time = start_time or time.time()
        self.end_time = end_time
        self.messages = messages or []
        self.completed_event = asyncio.Event()
        if completed:
            self.completed_event.set()

    def export_interactions(
        self, discount: float, style: str
    ) -> dict[str, "InteractionWithTokenLogpReward"]:
        if len(self.completions) == 0:
            return {}
        self.completions.apply_reward_discount(turn_discount=discount)
        return self.completions.export_interactions(style=style)


@dataclass
class SharedData:
    client: ArealOpenAI | None = field(default=None)
    session_cache: dict[str, SessionData] = field(default_factory=dict)
    session_ids_queue: asyncio.Queue | None = field(default=None)
    active_tasks: dict[str, bool] = field(default_factory=dict)
    num_active_tasks: int = field(default=0)
    task_sessions: dict[str, list[str]] = field(default_factory=dict)
    active_tasks_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def build_app(
    client: ArealOpenAI | None = None,
    session_cache: dict[str, SessionData] | None = None,
    session_ids_queue: asyncio.Queue | None = None,
    limit_active_tasks: bool = True,
    logger: logging.Logger | None = None,
):
    app = FastAPI()

    session_cache = session_cache if session_cache is not None else {}
    app.state.shared_data = SharedData(
        client=client,
        session_cache=session_cache,
        session_ids_queue=session_ids_queue,
        active_tasks=dict(),
        num_active_tasks=0,
        task_sessions=dict(),
        active_tasks_lock=asyncio.Lock(),
    )

    def get_shared_data() -> SharedData:
        return app.state.shared_data

    def get_client() -> ArealOpenAI:
        state = get_shared_data()
        if state.client is None:
            raise HTTPException(status_code=500, detail="Client not found")
        return state.client

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post(f"/v1/{RL_FINISH_TASK_PATHNAME}")
    async def finish_task(request: AReaLFinishTaskRequest):
        state = get_shared_data()
        task_id = request.task_id
        put_to_queue = request.put_to_queue

        async with state.active_tasks_lock:
            if task_id not in state.active_tasks:
                raise HTTPException(status_code=400, detail="Task not found")
            if state.active_tasks[task_id]:
                state.active_tasks[task_id] = False
                state.num_active_tasks -= 1
            else:
                raise HTTPException(status_code=400, detail="Task is already finished")

        if put_to_queue:
            queue = state.session_ids_queue
            if queue is None:
                raise HTTPException(status_code=500, detail="Queue not found")
            if queue.full():
                raise HTTPException(status_code=503, detail="Queue is full")
            await queue.put(state.task_sessions[task_id])
        return {"message": "success"}

    @app.post(f"/v1/{RL_START_SESSION_PATHNAME}")
    async def start_session(request: AReaLStartSessionRequest):
        """Start a new session or initialize from an existing session."""
        task_id = request.task_id
        init_from_session_id = request.init_from_session_id

        state = get_shared_data()

        async with state.active_tasks_lock:
            if task_id not in state.active_tasks:
                if limit_active_tasks:
                    workflow_executor = state.client.engine.workflow_executor
                    # check for capacity, should not have more active tasks than the running workflows
                    capacity = workflow_executor.staleness_manager.get_stats().running
                    if state.num_active_tasks >= capacity:
                        raise HTTPException(
                            status_code=429, detail="Too many active tasks"
                        )
                state.active_tasks[task_id] = True
                state.num_active_tasks += 1
                state.task_sessions[task_id] = []
            elif not state.active_tasks[task_id]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Task {task_id} is already finished, are you using same task_id multiple times?",
                )

            session_id = f"{task_id}-{len(state.task_sessions[task_id])}"
            state.task_sessions[task_id].append(session_id)

        session_cache = state.session_cache

        if init_from_session_id is not None:
            if init_from_session_id not in session_cache:
                raise HTTPException(status_code=400, detail="Session not found")
            session_cache[session_id] = deepcopy(session_cache[init_from_session_id])
            return {"session_id": session_id}

        session_cache[session_id] = SessionData(
            id=session_id,
            completed=False,
            completions=InteractionCache(),
            start_time=time.time(),
            end_time=None,
            messages=[],
        )
        return {"session_id": session_id}

    @app.post(f"/v1/{{session_id}}/{RL_END_SESSION_PATHNAME}")
    async def end_session(session_id: str):
        state = get_shared_data()
        if session_id not in state.session_cache:
            raise HTTPException(status_code=400, detail="Session not found")
        state.session_cache[session_id].completed = True
        state.session_cache[session_id].completed_event.set()
        return {"message": "success"}

    @app.post(f"/v1/{{session_id}}/{RL_SET_REWARD_PATHNAME}")
    async def set_reward(request: AReaLSetRewardRequest, session_id: str):
        interaction_id = request.interaction_id
        reward = request.reward

        state = get_shared_data()
        if session_id not in state.session_cache:
            raise HTTPException(
                status_code=400, detail=f"Session {session_id} not found"
            )
        if interaction_id is None:
            # take the last interaction id
            interaction_id = state.session_cache[
                session_id
            ].completions.last_interaction_id

        completions = state.session_cache[session_id].completions
        if interaction_id not in completions:
            raise HTTPException(
                status_code=400, detail=f"Interaction {interaction_id} not found"
            )
        state.session_cache[session_id].completions.set_reward(interaction_id, reward)
        return {"message": "success"}

    async def _call_client_create(
        create_fn: Callable[..., Awaitable[ChatCompletion | Response]],
        request: TypedDict | BaseModel,
        session_id: str,
        extra_ignored_args: list[str] | None = None,
    ) -> ChatCompletion | Response:
        state = get_shared_data()
        session_cache = state.session_cache
        if session_id not in session_cache:
            raise HTTPException(
                status_code=400, detail=f"Session {session_id} not found"
            )

        sig = inspect.signature(create_fn)
        areal_client_ignored_args = ["model"] + (extra_ignored_args or [])
        areal_client_disallowed_args = ["areal_cache"]
        areal_client_allowed_args = list(
            k
            for k in sig.parameters.keys()
            if k not in areal_client_ignored_args
            and k not in areal_client_disallowed_args
        )

        kwargs = (
            request.model_dump() if isinstance(request, BaseModel) else dict(request)
        )
        dropped_args = []
        for k, v in kwargs.items():
            if k not in areal_client_allowed_args:
                dropped_args.append((k, v))

        for k, _ in dropped_args:
            del kwargs[k]

        def _is_default_value(k: str, v: Any) -> bool:
            if isinstance(request, BaseModel):
                return v == type(request).model_fields[k].default
            return False

        dropped_non_default_args = [
            (k, v)
            for k, v in dropped_args
            if k not in areal_client_ignored_args and not _is_default_value(k, v)
        ]
        if len(dropped_non_default_args):
            dropped_args_str = "\n".join(
                [f"  {k}: {v}" for k, v in dropped_non_default_args]
            )
            if logger is not None:
                logger.warning(
                    f"dropped unsupported non-default arguments for areal client:\n"
                    f"{dropped_args_str}"
                )

        if "temperature" not in kwargs:
            kwargs["temperature"] = 1.0
            if logger is not None:
                logger.warning("temperature not set in request, defaulting to 1.0")
        if "top_p" not in kwargs:
            kwargs["top_p"] = 1.0
            if logger is not None:
                logger.warning("top_p not set in request, defaulting to 1.0")

        try:
            return await create_fn(
                areal_cache=session_cache[session_id].completions, **kwargs
            )
        except ValueError as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post(
        f"/v1/{{session_id}}/{CHAT_COMPLETIONS_PATHNAME}",
        dependencies=[Depends(validate_json_request)],
    )
    async def chat_completions(
        request: CompletionCreateParams, session_id: str
    ) -> ChatCompletion:
        client = get_client()
        return await _call_client_create(
            create_fn=client.chat.completions.create,
            request=request,
            session_id=session_id,
        )

    @app.post(
        f"/v1/{{session_id}}/{RESPONSES_PATHNAME}",
        dependencies=[Depends(validate_json_request)],
    )
    async def responses(request: ResponseCreateParams, session_id: str) -> Response:
        client = get_client()
        return await _call_client_create(
            create_fn=client.responses.create,
            request=request,
            session_id=session_id,
            extra_ignored_args=["request_id"],
        )

    return app


class ProxyServer:
    def __init__(
        self,
        port: int | None = None,
        *,
        client: ArealOpenAI | None = None,
        name: str = "proxy server",
        session_cache: dict[str, SessionData] | None = None,
        buffer_size: int | None = None,  # buffer size for session ids queue
        limit_active_tasks: bool = True,
        server_log_level: int = logging.INFO,
        uvicorn_log_level: int = logging.WARNING,
    ):
        self.port = port if port is not None else find_free_ports(1)[0]
        self.client = client
        self.name = name
        self.session_cache = session_cache if session_cache is not None else {}
        self.session_ids_queue = asyncio.Queue(maxsize=buffer_size or 0)
        self.logger = getLogger("ArealOpenAI Proxy", level=server_log_level)
        self.app = build_app(
            client=client,
            session_cache=self.session_cache,
            session_ids_queue=self.session_ids_queue,
            limit_active_tasks=limit_active_tasks,
            logger=self.logger,
        )
        self.host_ip = gethostip()
        self._localhost = "0.0.0.0"
        self.server_config = uvicorn.Config(
            self.app, host=self._localhost, port=self.port, log_level=uvicorn_log_level
        )
        self.server = uvicorn.Server(self.server_config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    @property
    def public_addr(self) -> str:
        return f"http://{self.host_ip}:{self.port}"

    @property
    def local_addr(self) -> str:
        return f"http://{self._localhost}:{self.port}"

    def check_health(self, timeout: int = 20) -> bool:
        try:
            response = requests.get(f"{self.local_addr}/health", timeout=timeout)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def wait_until_ready(self, timeout: int = 20):
        while not self.check_health(timeout):
            self.logger.info(
                f"Waiting for {self.name} {self.public_addr} to be ready..."
            )
            time.sleep(1)
        self.logger.info(f"{self.name} {self.public_addr} is ready!")

    def start(self, wait_until_ready: bool = True, timeout: int = 20):
        self.logger.info(f"Starting {self.name} on {self.public_addr}")
        self.thread.start()
        if wait_until_ready:
            self.wait_until_ready(timeout)

    def close(self):
        self.server.should_exit = True
        if self.thread.is_alive():
            self.thread.join(timeout=5.0)  # Wait up to 5 seconds
            if self.thread.is_alive():
                self.logger.warning(
                    "Proxy server thread did not stop gracefully within timeout"
                )
        self.logger.info(f"{self.name} stopped.")

    async def get_session_ids(self) -> list[str]:
        session_ids = await self.session_ids_queue.get()
        return session_ids

    async def get_session_cache_data(
        self, session_id: str, ensure_completed: bool = True
    ) -> SessionData:
        if ensure_completed:
            # Wait for session to exist (should be quick, but handle race condition)
            while session_id not in self.session_cache:
                await asyncio.sleep(0.1)
            # Wait for session to be completed using event (non-blocking, efficient)
            await self.session_cache[session_id].completed_event.wait()
        return self.session_cache[session_id]

    async def get_results(
        self, session_ids: list[str], discount: float = 1.0, style: str = "individual"
    ) -> tuple[dict[str, float | None], dict[str, "InteractionWithTokenLogpReward"]]:
        session_caches = await asyncio.gather(
            *[self.get_session_cache_data(session_id) for session_id in session_ids]
        )
        rewards = dict(
            zip(
                session_ids,
                [result.completions.total_reward for result in session_caches],
            )
        )
        completions = {}
        for result in session_caches:
            completions.update(result.export_interactions(discount, style))
        return rewards, completions

    async def get_interactions(
        self, session_ids: list[str], discount: float = 1.0, style: str = "individual"
    ) -> dict[str, "InteractionWithTokenLogpReward"]:
        results = await self.get_results(session_ids, discount, style)
        return dict(zip(session_ids, [completion for _, completion in results.items()]))

    @classmethod
    async def finish_task(cls, task_id: str, base_url: str, put_to_queue: bool = True):
        async with aiohttp.ClientSession(
            ensure_end_with_slash(base_url)
        ) as http_session:
            await post_json_with_retry(
                http_session,
                RL_FINISH_TASK_PATHNAME,
                payload=AReaLFinishTaskRequest(
                    task_id=task_id, put_to_queue=put_to_queue
                ),
            )


class ProxySession(aiohttp.ClientSession):
    def __init__(self, base_url: str, task_id: str, *args, **kwargs):
        base_url = ensure_end_with_slash(base_url)
        super().__init__(base_url, *args, **kwargs)

        self.base_url = base_url
        self.task_id = task_id
        self.session_id = None

    @property
    def session_url(self) -> str:
        return str(self._build_url(self.session_id))

    async def set_reward(self, completion_id: str, reward: float):
        """Set reward for a specific completion/response by its ID."""
        if self.session_id is None:
            raise ValueError("Session ID is not set")
        await set_interaction_reward(
            self,
            interaction_id=completion_id,
            reward=reward,
            url=f"{self.session_id}/{RL_SET_REWARD_PATHNAME}",
        )

    async def set_last_reward(self, reward: float):
        """Set reward for the most recent completion/response."""
        if self.session_id is None:
            raise ValueError("Session ID is not set")
        await set_last_interaction_reward(
            self, reward=reward, url=f"{self.session_id}/{RL_SET_REWARD_PATHNAME}"
        )

    async def __aenter__(self) -> "ProxySession":
        await super().__aenter__()
        data = await _start_session(
            self,
            RL_START_SESSION_PATHNAME,
            payload=AReaLStartSessionRequest(task_id=self.task_id),
        )
        self.session_id = data["session_id"]
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ):
        if exc_type is not None:
            await super().__aexit__(exc_type, exc_val, exc_tb)
            return

        if self.session_id is None:
            raise ValueError("Session ID is not set")

        await post_json_with_retry(self, f"{self.session_id}/{RL_END_SESSION_PATHNAME}")
        await super().__aexit__(exc_type, exc_val, exc_tb)


if __name__ == "__main__":
    server = ProxyServer()
    server.start(wait_until_ready=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.close()
