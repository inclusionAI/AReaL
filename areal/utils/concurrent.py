import asyncio
import concurrent.futures


def run_async_task(func, *args, **kwargs):
    # Check if we're already in an async context
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None:
        # Already in async context, create a new task
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, func(*args, **kwargs))
            return future.result()
    else:
        # Not in async context, use asyncio.run directly
        return asyncio.run(func(*args, **kwargs))
