"""
Utilities for handling async operations in thread-safe manner.

This module provides utilities to safely run async code in contexts where
there might not be an event loop available (e.g., ThreadPoolExecutor threads).
"""

import asyncio
import threading
from typing import Any, Awaitable, Callable


class AsyncRunner:
    """Thread-safe async runner utility."""
    
    @staticmethod
    def run(coro: Awaitable[Any]) -> Any:
        """
        Run an async coroutine in a thread-safe manner.
        
        This method handles the case where there might not be an event loop
        in the current thread (common in ThreadPoolExecutor threads).
        
        Args:
            coro: The coroutine to run
            
        Returns:
            The result of the coroutine
            
        Raises:
            Exception: Any exception raised by the coroutine
        """
        try:
            # Try to get the current event loop
            try:
                loop = asyncio.get_running_loop()
                # If we're already in an event loop, use it
                if loop.is_running():
                    # Create a task and wait for it
                    return asyncio.create_task(coro)
                else:
                    # Event loop exists but not running, use it
                    return loop.run_until_complete(coro)
            except RuntimeError:
                # No event loop in current thread
                pass
                
            # Check if we're in the main thread
            if threading.current_thread() is threading.main_thread():
                # In main thread, use asyncio.run
                return asyncio.run(coro)
            else:
                # In worker thread, create new event loop
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    return loop.run_until_complete(coro)
                finally:
                    try:
                        loop.close()
                    except Exception:
                        pass
                        
        except Exception as e:
            # Fallback to asyncio.run if all else fails
            return asyncio.run(coro)
    
    @staticmethod
    def run_with_loop(coro: Awaitable[Any]) -> Any:
        """
        Run an async coroutine, ensuring an event loop is available.
        
        This is a simpler version that just ensures an event loop exists.
        
        Args:
            coro: The coroutine to run
            
        Returns:
            The result of the coroutine
        """
        # 始终创建新的事件循环，避免线程间冲突
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            try:
                loop.close()
            except Exception:
                pass


def run_async(coro: Awaitable[Any]) -> Any:
    """
    Convenience function to run an async coroutine in a thread-safe manner.
    
    Args:
        coro: The coroutine to run
        
    Returns:
        The result of the coroutine
    """
    return AsyncRunner.run(coro)


def run_async_with_loop(coro: Awaitable[Any]) -> Any:
    """
    Convenience function to run an async coroutine with event loop handling.
    
    Args:
        coro: The coroutine to run
        
    Returns:
        The result of the coroutine
    """
    return AsyncRunner.run_with_loop(coro)