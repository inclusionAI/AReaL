"""Generic asynchronous task runner for executing concurrent async Python functions.

This module provides a reusable, thread-based async task executor that can run
any async Python functions concurrently with queue management, pause/resume control,
and health monitoring. It has no dependencies on AReaL-specific logic.

The AsyncTaskRunner manages a background thread running an asyncio event loop (uvloop)
that processes tasks from an input queue and places results in an output queue.
"""

from __future__ import annotations

import asyncio
import queue
import random
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

import uvloop


# Type variable for generic result types
T = TypeVar("T")

# Polling configuration
DEFAULT_POLL_WAIT_TIME = 0.05  # 50ms
DEFAULT_POLL_SLEEP_TIME = 0.5  # 1 second


class TaskQueueFullError(RuntimeError):
    """Raised when an AsyncTaskRunner queue is full."""


@dataclass
class _TimedResult(Generic[T]):
    """Internal wrapper for results with creation timestamp."""

    create_time: int  # nanoseconds from time.monotonic_ns()
    data: T


@dataclass
class _TaskInput(Generic[T]):
    """Internal wrapper for task input with async function and arguments."""

    async_fn: Callable[..., Awaitable[T]]
    args: tuple
    kwargs: dict


@dataclass
class _Task(Generic[T]):
    """Internal wrapper for running task with metadata."""

    create_time: int  # nanoseconds from time.monotonic_ns()
    task: asyncio.Task
    task_input: _TaskInput[T]


class AsyncTaskRunner(Generic[T]):
    """Generic asynchronous task runner with queue management and pause/resume control.

    This class provides a reusable async task executor that runs a background thread
    with an asyncio event loop (using uvloop for performance). It can execute any
    async Python function concurrently with configurable queue sizes and optional
    pause/resume control.

    The runner maintains thread-safe input and output queues, manages task lifecycle,
    and provides health monitoring to detect thread failures.

    Parameters
    ----------
    max_queue_size : int
        Maximum size for input and output queues. Tasks submitted when
        the input queue is full will raise RuntimeError.
    poll_wait_time : float, optional
        Time in seconds to wait for task completion during each poll
        cycle. Default is 0.05 (50ms).
    poll_sleep_time : float, optional
        Time in seconds to sleep between poll cycles.
        Default is 1.0 second.
    enable_tracing : bool, optional
        Enable detailed logging of task submission and completion.
        Default is False.

    Attributes
    ----------
    input_queue : queue.Queue
        Thread-safe queue for incoming task submissions.
    output_queue : queue.Queue
        Thread-safe queue for completed task results.
    exiting : threading.Event
        Signal to request thread shutdown.
    paused : threading.Event
        Signal to pause new task creation (existing tasks continue).

    Examples
    --------
    Basic usage with simple async functions:

    >>> import asyncio
    >>> runner = AsyncTaskRunner[int](max_queue_size=100)
    >>> runner.initialize()
    >>>
    >>> async def compute(x: int) -> int:
    ...     await asyncio.sleep(0.1)
    ...     return x * 2
    >>>
    >>> # Submit tasks
    >>> for i in range(5):
    ...     runner.submit(compute, i)
    >>>
    >>> # Wait for results
    >>> results = runner.wait(count=5)
    >>> print(results)  # [0, 2, 4, 6, 8] (order may vary)
    >>>
    >>> runner.destroy()

    Using pause/resume for control:

    >>> runner = AsyncTaskRunner[str](max_queue_size=50)
    >>> runner.initialize()
    >>>
    >>> async def fetch_data(url: str) -> str:
    ...     # Simulate network request
    ...     await asyncio.sleep(0.5)
    ...     return f"Data from {url}"
    >>>
    >>> # Submit some tasks
    >>> for i in range(10):
    ...     runner.submit(fetch_data, f"http://example.com/{i}")
    >>>
    >>> # Pause to prevent new tasks from starting
    >>> runner.pause()
    >>>
    >>> # Wait for currently running tasks
    >>> results = runner.wait(count=5, timeout=2.0)
    >>>
    >>> # Resume and submit more
    >>> runner.resume()
    >>> runner.destroy()

    See Also
    --------
    WorkflowExecutor : AReaL-specific wrapper that adds staleness management
    """

    def __init__(
        self,
        max_queue_size: int,
        poll_wait_time: float = DEFAULT_POLL_WAIT_TIME,
        poll_sleep_time: float = DEFAULT_POLL_SLEEP_TIME,
        enable_tracing: bool = False,
    ):
        """Initialize the AsyncTaskRunner.

        Parameters
        ----------
        max_queue_size : int
            Maximum size for input and output queues.
        poll_wait_time : float, optional
            Time in seconds to wait for task completion during polling.
            Default is 0.05.
        poll_sleep_time : float, optional
            Time in seconds to sleep between poll cycles.
            Default is 1.0.
        enable_tracing : bool, optional
            Enable detailed logging. Default is False.
        """
        self.max_queue_size = max_queue_size
        self.poll_wait_time = poll_wait_time
        self.poll_sleep_time = poll_sleep_time
        self.enable_tracing = enable_tracing

        # Thread control
        self.exiting = threading.Event()
        self.paused = threading.Event()

        # Queues for task management
        self.input_queue: queue.Queue[_TaskInput[T]] = queue.Queue(
            maxsize=max_queue_size
        )
        self.output_queue: queue.Queue[_TimedResult[T]] = queue.Queue(
            maxsize=max_queue_size
        )

        # Cache for results to support wait() with arbitrary counts
        self.result_cache: list[_TimedResult[T]] = []

        # Thread exception handling
        self._thread_exception_lock = threading.Lock()
        self._thread_exception: Exception | None = None

        # Will be set in initialize()
        self.logger = None
        self.thread: threading.Thread | None = None

    def initialize(self, logger=None):
        """Initialize and start the background thread.

        This method starts the background thread that runs the asyncio
        event loop. Must be called before submitting any tasks.

        Parameters
        ----------
        logger : logging.Logger, optional
            Logger instance for debugging and tracing.
            If None, logging is minimal.
        """
        self.logger = logger

        # Start the background thread (daemon=True for automatic cleanup)
        self.thread = threading.Thread(target=self._run_thread, daemon=True)
        self.thread.start()

    def destroy(self):
        """Shutdown the task runner and wait for thread cleanup.

        This method signals the background thread to exit and waits for
        it to complete. All pending tasks will be cancelled.
        """
        self.exiting.set()
        if self.thread is not None:
            self.thread.join()

    def _check_thread_health(self):
        """Check if the background thread has encountered a fatal error.

        Raises
        ------
        RuntimeError
            If the background thread has died due to an exception.
        """
        with self._thread_exception_lock:
            if self._thread_exception is not None:
                raise RuntimeError(
                    "AsyncTaskRunner thread has died due to an exception. "
                    "No further tasks can be processed."
                ) from self._thread_exception

    def _run_thread(self):
        """Entry point for the background thread.

        Runs the async event loop and handles exceptions.
        """
        try:
            uvloop.run(self._run_async_loop())
        except Exception as e:
            # Store exception for thread-safe access
            with self._thread_exception_lock:
                self._thread_exception = e
            if self.logger:
                self.logger.error(
                    f"AsyncTaskRunner thread failed with exception: {e}",
                    exc_info=True,
                )
            # Signal that we're exiting due to error
            self.exiting.set()

    async def _run_async_loop(self):
        """Main async event loop that processes tasks.

        This loop:
        1. Pulls tasks from input_queue when not paused
        2. Creates asyncio.Task instances for each
        3. Waits for task completion
        4. Places results in output_queue
        5. Continues until exiting signal is set
        """
        running_tasks: dict[str, _Task[T]] = {}
        task_id = 0

        try:
            while not self.exiting.is_set():
                # Pull new tasks from input queue when not paused
                while not self.paused.is_set() and self.input_queue.qsize() > 0:
                    try:
                        task_input = self.input_queue.get_nowait()
                        task_input: _TaskInput[T]

                        # Create asyncio task
                        async_task = asyncio.create_task(
                            task_input.async_fn(*task_input.args, **task_input.kwargs),
                            name=str(task_id),
                        )

                        # Store task with metadata
                        running_tasks[str(task_id)] = _Task(
                            create_time=time.monotonic_ns(),
                            task=async_task,
                            task_input=task_input,
                        )

                        if self.enable_tracing and self.logger:
                            self.logger.info(
                                f"AsyncTaskRunner: Submitted task {task_id}. "
                                f"Running: {len(running_tasks)}"
                            )

                        task_id += 1
                    except queue.Empty:
                        break

                # Wait for any task to complete
                done = []
                if running_tasks:
                    tasks = [t.task for t in running_tasks.values()]
                    done, _ = await asyncio.wait(
                        tasks,
                        timeout=self.poll_wait_time,
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                # Process completed tasks
                for async_task in done:
                    tid = async_task.get_name()
                    task_obj = running_tasks.pop(tid)
                    try:
                        result = await async_task
                    except asyncio.CancelledError:
                        if self.logger:
                            self.logger.warning(
                                f"Task {tid} was cancelled. None will be returned"
                            )
                        result = None
                    except Exception as e:
                        if self.logger:
                            self.logger.error(
                                f"AsyncTaskRunner: Task {tid} "
                                f"failed with exception: {e} ",
                                exc_info=True,
                            )
                        result = None

                    try:
                        # Place result in output queue
                        self.output_queue.put_nowait(
                            _TimedResult(create_time=task_obj.create_time, data=result)
                        )
                        if self.enable_tracing and self.logger:
                            self.logger.info(
                                f"AsyncTaskRunner: Completed task {tid}. "
                                f"Running: {len(running_tasks)}"
                            )
                    except queue.Full:
                        # This is a critical error that should stop the runner.
                        # Re-add task so it can be cancelled in finally.
                        running_tasks[tid] = task_obj
                        if self.logger:
                            self.logger.critical(
                                f"Output queue is full. Task ID: {tid}. "
                                f"Please increase max_queue_size.",
                                exc_info=True,
                            )
                        raise TaskQueueFullError(
                            "Output queue full. Please increase max_queue_size."
                        )
                await asyncio.sleep(self.poll_sleep_time)
        finally:
            # Cancel all remaining tasks on shutdown
            pending_tasks = [
                task_obj.task
                for task_obj in running_tasks.values()
                if not task_obj.task.done()
            ]
            if pending_tasks:
                for task in pending_tasks:
                    task.cancel()
                await asyncio.gather(*pending_tasks, return_exceptions=True)

    def submit(
        self,
        async_fn: Callable[..., Awaitable[T]],
        *args,
        **kwargs,
    ) -> None:
        """Submit an async function for execution.

        The function will be executed in the background thread's event
        loop. Results can be retrieved using wait().

        Parameters
        ----------
        async_fn : Callable[..., Awaitable[T]]
            The async function to execute.
        *args
            Positional arguments to pass to the function.
        **kwargs
            Keyword arguments to pass to the function.

        Raises
        ------
        RuntimeError
            If the input queue is full or if the background thread
            has died.

        Examples
        --------
        >>> async def add(a: int, b: int) -> int:
        ...     return a + b
        >>>
        >>> runner.submit(add, 5, 10)
        >>> runner.submit(add, a=3, b=7)
        """
        # Check if thread is still alive
        self._check_thread_health()

        # Create task input wrapper
        task_input = _TaskInput(async_fn=async_fn, args=args, kwargs=kwargs)

        # Submit to queue
        try:
            self.input_queue.put_nowait(task_input)
        except queue.Full:
            raise TaskQueueFullError(
                "Input queue full. Please increase max_queue_size or "
                "wait for tasks to complete."
            )

    def wait(self, count: int, timeout: float | None = None) -> list[T]:
        """Wait for a specified number of task results.

        This method blocks until at least `count` results are available
        or the timeout expires. Results are returned in random order
        (shuffled).

        Parameters
        ----------
        count : int
            Number of results to wait for.
        timeout : float | None, optional
            Maximum time in seconds to wait. If None, waits indefinitely
            (up to 7 days). Default is None.

        Returns
        -------
        List[T]
            List of task results, shuffled randomly.

        Raises
        ------
        TimeoutError
            If timeout expires before `count` results are available.
        RuntimeError
            If the background thread exits before results are ready.

        Examples
        --------
        >>> runner.submit(compute, 1)
        >>> runner.submit(compute, 2)
        >>> runner.submit(compute, 3)
        >>> results = runner.wait(count=3, timeout=10.0)
        >>> len(results)
        3
        """
        start_time = time.perf_counter()
        timeout = timeout or float(7 * 24 * 3600)  # 7 days default

        while not self.exiting.is_set() and time.perf_counter() - start_time < timeout:
            # Check thread health
            self._check_thread_health()

            # Drain all available results from output queue
            while True:
                try:
                    timed_result = self.output_queue.get_nowait()
                    self.result_cache.append(timed_result)
                except queue.Empty:
                    break

            # Check if we have enough results
            if len(self.result_cache) >= count:
                break

            # Sleep briefly to avoid busy waiting
            time.sleep(self.poll_sleep_time)

        # Check exit conditions
        if self.exiting.is_set():
            self._check_thread_health()
            raise RuntimeError("AsyncTaskRunner is exiting, cannot wait for results.")

        accepted = len(self.result_cache)
        if accepted < count:
            raise TimeoutError(
                f"Timed out waiting for {count} results, only received {accepted}."
            )

        # Sort by creation time for deterministic ordering
        self.result_cache.sort(key=lambda x: x.create_time)

        # Extract the requested number of results
        results_to_return = self.result_cache[:count]
        self.result_cache = self.result_cache[count:]

        # Shuffle for randomness (helps with data diversity in ML)
        random.shuffle(results_to_return)

        # Extract just the data (remove timing metadata)
        return [r.data for r in results_to_return]

    def submit_batch(
        self,
        tasks: list[tuple[Callable[..., Awaitable[T]], tuple, dict]],
    ) -> None:
        """Submit multiple tasks at once.

        Convenience method for submitting multiple tasks in a single call.

        Parameters
        ----------
        tasks : List[tuple[Callable, tuple, dict]]
            List of (async_fn, args, kwargs) tuples to submit.

        Examples
        --------
        >>> tasks = [
        ...     (compute, (1,), {}),
        ...     (compute, (2,), {}),
        ...     (compute, (3,), {}),
        ... ]
        >>> runner.submit_batch(tasks)
        """
        for async_fn, args, kwargs in tasks:
            self.submit(async_fn, *args, **kwargs)

    def pause(self):
        """Pause submission of new tasks.

        After calling pause(), no new tasks will be started from the
        input queue, but existing running tasks will continue to
        completion.
        """
        self.paused.set()

    def resume(self):
        """Resume submission of new tasks.

        Allows new tasks to be pulled from the input queue and
        started.
        """
        self.paused.clear()

    def get_queue_sizes(self) -> tuple[int, int]:
        """Get current sizes of input and output queues.

        Returns
        -------
        tuple[int, int]
            (input_queue_size, output_queue_size)
        """
        return self.input_queue.qsize(), self.output_queue.qsize()

    def get_input_queue_size(self) -> int:
        """Get current size of the input queue.

        Returns
        -------
        int
            Number of tasks waiting in the input queue.
        """
        return self.input_queue.qsize()

    def get_output_queue_size(self) -> int:
        """Get current size of the output queue.

        Returns
        -------
        int
            Number of completed results waiting in the output queue.
        """
        return self.output_queue.qsize()
