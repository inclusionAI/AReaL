import asyncio
import atexit
import concurrent.futures
import threading

# Lazy-initialized shared executor to reduce thread creation/destruction overhead
# when run_async_task is called frequently
_shared_executor: concurrent.futures.ThreadPoolExecutor | None = None
_executor_lock = threading.Lock()


def get_executor() -> concurrent.futures.ThreadPoolExecutor:
    """Get or create the shared thread pool executor.

    This provides a global shared executor for background tasks that need
    to run in separate threads. The executor is lazily initialized and
    automatically cleaned up at process exit.

    Returns
    -------
    ThreadPoolExecutor
        A shared thread pool executor with 4 workers.
    """
    global _shared_executor
    if _shared_executor is None:
        with _executor_lock:
            if _shared_executor is None:
                _shared_executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=4, thread_name_prefix="shared_executor"
                )
                # Register cleanup on process exit
                atexit.register(_shutdown_executor)
    return _shared_executor


def _shutdown_executor() -> None:
    """Shutdown the shared thread pool executor if it exists.

    Called via atexit at process exit, when no other threads should be
    accessing the executor.
    """
    global _shared_executor
    if _shared_executor is not None:
        _shared_executor.shutdown(wait=False)
        _shared_executor = None


def run_async_task(func, *args, **kwargs):
    # Check if we're already in an async context
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None:
        # Already in async context, use shared executor to avoid overhead
        future = get_executor().submit(asyncio.run, func(*args, **kwargs))
        return future.result()
    else:
        # Not in async context, use asyncio.run directly
        return asyncio.run(func(*args, **kwargs))
