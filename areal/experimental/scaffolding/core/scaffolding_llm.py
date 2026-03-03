# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# Vendored from tensorrt_llm.scaffolding.scaffolding_llm

import asyncio
import threading
import traceback
from collections import deque
from collections.abc import Generator, Mapping
from dataclasses import dataclass
from typing import Any

from areal.utils import logging as areal_logging

from .controller import Controller, ParallelProcess
from .result import ScaffoldingResult
from .task import Task
from .worker import Worker

_logger = areal_logging.getLogger("ScaffoldingLlm")


@dataclass(frozen=True)
class ScaffoldingRequest:
    prompt: str
    kwargs: Mapping[str, Any]
    controller: Controller
    result: "ScaffoldingResult"


class ScaffoldingLlm:
    def __init__(
        self,
        prototype_controller: Controller,
        workers: Mapping[str, Worker],  # map of role to worker instance,
        max_parallel_requests: int = 64,
    ):
        self.prototype_controller = prototype_controller
        self.workers = workers

        self.loop = self._get_loop()
        asyncio.set_event_loop(self.loop)
        self.task_queue = asyncio.Queue()
        self.main_loop_stop_event = asyncio.Event()
        self.shutdown_event = asyncio.Event()
        if self.own_loop:
            self._run_main_loop_thread()
        else:
            self._run_main_loop_coroutine()

        # For top scheduler
        self.running_req_count = 0
        self.max_parallel_requests = max_parallel_requests
        self.pending_queue = deque()

        self.output_task_collection = False

    def __enter__(self):
        return self

    def __exit__(self):
        self.shutdown()

    def _get_loop(self):
        try:
            self.own_loop = False
            return asyncio.get_running_loop()
        except RuntimeError:
            self.own_loop = True
            return asyncio.new_event_loop()
        return None

    def _schedule_on_loop(self, coro):
        """Schedule a coroutine on self.loop.

        Uses create_task when called from within the event loop thread
        (to avoid deadlocks with run_coroutine_threadsafe on uvloop),
        and falls back to run_coroutine_threadsafe for cross-thread calls.
        """
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None

        if running is self.loop:
            _logger.info("_schedule_on_loop: same loop, create_task for %s", coro.__name__)
            self.loop.create_task(coro)
        elif running is not None:
            _logger.warning(
                "_schedule_on_loop: different loop! running=%s self.loop=%s",
                id(running), id(self.loop),
            )
            asyncio.run_coroutine_threadsafe(coro, self.loop)
        else:
            _logger.info("_schedule_on_loop: no running loop, run_coroutine_threadsafe for %s", coro.__name__)
            asyncio.run_coroutine_threadsafe(coro, self.loop)

    async def _handle_controller_generator(
        self, gen: Generator, request: ScaffoldingRequest = None
    ):
        """Handle a controller generator, processing tasks and parallel processes."""
        step = 0
        for obj in gen:
            if isinstance(obj, ParallelProcess):
                await self._handle_parallel_process(obj, request)
            else:
                step += 1
                _logger.info(
                    "Dispatching task list step=%d, tasks=%d, types=%s",
                    step, len(obj), [type(t).__name__ for t in obj],
                )
                await self._handle_task_list(obj, request)
                _logger.info("Task list step=%d completed.", step)

    async def _handle_task_list(
        self, tasks: list[Task], request: ScaffoldingRequest = None
    ):
        """Execute a list of tasks concurrently."""
        async_tasks = [
            asyncio.create_task(self.workers[task.worker_tag].run_task(task))
            for task in tasks
        ]
        await asyncio.gather(*async_tasks)
        for task in tasks:
            if task.streaming_output_flag:
                for output in task.streaming_output_list:
                    request.result.set_output_streaming(output)
                task.streaming_output_list = []

    async def _handle_parallel_process(
        self, tasks: ParallelProcess, request: ScaffoldingRequest = None
    ):
        """Handle parallel execution of multiple generators."""
        async_tasks = [
            asyncio.create_task(self._handle_controller_generator(sub_gen, request))
            for sub_gen in tasks.sub_gens
        ]
        await asyncio.gather(*async_tasks)

    async def _handle_single_request(self, request: ScaffoldingRequest):
        """Process a single scaffolding request."""
        try:
            gen = self._create_controller_generator(request)
            await self._handle_controller_generator(gen, request)
        except Exception as e:
            print(f"ScaffoldingLLM request exception: {e}")
            traceback.print_exc()
            request.result.set_output(None)
            raise
        finally:
            self.running_req_count -= 1
            self._maybe_schedule()

    def _create_controller_generator(self, request: ScaffoldingRequest):
        """Create a generator wrapper for the controller."""
        scaffolding_output = yield from request.controller.generate(
            request.prompt, **request.kwargs
        )

        if self.output_task_collection:
            request.result.set_task_collections(request.controller.task_collections)
        request.result.set_output(scaffolding_output)

    def _schedule_request(self, request: ScaffoldingRequest):
        """Schedule a single request for execution."""
        asyncio.create_task(self._handle_single_request(request))
        self.running_req_count += 1

    def _maybe_schedule(self, request: ScaffoldingRequest = None):
        """Schedule pending requests if capacity allows."""
        if self.shutdown_event.is_set():
            return

        if request is not None:
            self.pending_queue.append(request)

        while (
            self.running_req_count < self.max_parallel_requests and self.pending_queue
        ):
            next_request = self.pending_queue.popleft()
            self._schedule_request(next_request)

    async def _handle_event_loop(self):
        """Main event handling loop."""
        while True:
            item = await self.task_queue.get()

            if item is None:
                return
            elif isinstance(item, ScaffoldingRequest):
                self._maybe_schedule(item)
            else:
                raise ValueError(f"Unsupported task_queue item type: {type(item)}")

    async def _main_loop_async_func(self):
        """Main async loop function."""
        _logger.info("_main_loop_async_func STARTED (loop=%s)", id(asyncio.get_running_loop()))
        handle_event_task = asyncio.create_task(self._handle_event_loop())
        await handle_event_task
        self.main_loop_stop_event.set()

    def _run_main_loop_coroutine(self):
        self._schedule_on_loop(self._main_loop_async_func())

    def _run_main_loop_thread(self):
        def main_loop_thread():
            self.loop.run_until_complete(self._main_loop_async_func())

        self.main_loop_thread = threading.Thread(target=main_loop_thread)
        self.main_loop_thread.start()

    def generate_async(self, prompt: str) -> ScaffoldingResult:
        result = ScaffoldingResult()

        async def put_request():
            try:
                request = ScaffoldingRequest(
                    prompt=prompt,
                    kwargs={},
                    result=result,
                    controller=self.prototype_controller.clone(),
                )
            except Exception as e:
                self.task_queue.put(None)
                print(
                    f"Error: build ScaffoldingRequest failed: {e} \n {traceback.format_exc()}"
                )
            else:
                await self.task_queue.put(request)

        self._schedule_on_loop(put_request())

        return result

    def generate(
        self, prompts: str | list[str]
    ) -> ScaffoldingResult | list[ScaffoldingResult]:
        unbatched = not isinstance(prompts, list)
        batched_prompts = [prompts] if unbatched else prompts

        scaffolding_results = []
        for prompt in batched_prompts:
            scaffolding_results.append(self.generate_async(prompt))

        for scaffolding_result in scaffolding_results:
            scaffolding_result.result()

        return scaffolding_results[0] if unbatched else scaffolding_results

    def enable_output_task_collection(self):
        self.output_task_collection = True

    def shutdown(self, shutdown_workers=False):
        def shutdown_workers_func():
            for worker in self.workers.values():
                worker.shutdown()

        async def stop_task_on_loop():
            await self.task_queue.put(None)
            await self.main_loop_stop_event.wait()
            for worker in self.workers.values():
                await worker.async_shutdown()

        self._schedule_on_loop(stop_task_on_loop())

        if self.own_loop:
            self.main_loop_thread.join()
        else:
            self.shutdown_event.set()

        if shutdown_workers:
            shutdown_workers_func()
