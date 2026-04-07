import IPython
import threading
import time
import gc
import psutil
import sys
import os
import site
import asyncio
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr
from collections import OrderedDict
from typing import Dict, Optional, Tuple
import logging
from IPython.terminal.interactiveshell import TerminalInteractiveShell, InteractiveShell
from pathlib import Path
import ray
from ray.exceptions import GetTimeoutError
from traitlets.config import Config

import signal
import threading
from contextlib import contextmanager


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
site_packages_dir = site.getsitepackages()[0]
file_abs_path = os.path.abspath(__file__)

# ==== Ray + Actor-based kernel management =====================================

# How many actors (kernels) we cache
MAX_ACTORS = 1024


def _ensure_ray_initialized():
    """Initialize Ray once (in local mode by default)."""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False)
        logger.info("Ray initialized")

_ensure_ray_initialized()


class TimeoutException(Exception):
    """Raised when execution times out"""
    pass


@contextmanager
def time_limit(seconds):
    """Context manager that raises TimeoutException after specified seconds"""
    def signal_handler(signum, frame):
        raise TimeoutException(f"Execution time out after {seconds} seconds")
    
    old_handler = signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


@ray.remote(num_cpus=0)
class KernelActor:
    """
    A Ray actor that owns a single IPython shell in its own process.
    """

    def __init__(self):
        c = Config()
        c.HistoryManager.enabled = False
        shell = InteractiveShell.instance(colors='NoColor', config=c)
        shell.history_manager.enabled = False
        shell.cache_size = 1000

        try:    
            import matplotlib
            matplotlib.use('Agg')
        except Exception:
            pass

        self.shell = shell
        self.actor_id = ray.get_runtime_context().get_actor_id()
        logger.info(f"KernelActor initialized with actor_id={self.actor_id}")

    def get_actor_id(self) -> str:
        """Return the Ray actor ID for debugging."""
        return self.actor_id

    def execute(self, script: str, max_output_size: int = 100 * 1024, timeout: int = 120) -> Tuple[str, bool]:
        """
        Run a script in this actor's IPython shell, capturing stdout/stderr.
        """
        if not isinstance(script, str) or not script.strip():
            return "Error: script must be a non-empty string", False

        stdout_capture = StringIO()
        stderr_capture = StringIO()
        success = True
        
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            try:
                with time_limit(timeout):
                    result = self.shell.run_cell(script, silent=False)
                    
                if result.error_before_exec:
                    error_output = f"Error before execution: {result.error_before_exec}"
                    success = False
                elif result.error_in_exec:
                    error_output = f"Error during execution: {result.error_in_exec}"
                    success = False
                else:
                    error_output = ""
                    success = True
            except TimeoutException as e:
                error_output = f"Execution time out after {timeout} seconds"
                success = False
            except Exception as e:
                error_output = f"Execution error: {str(e)}"
                success = False

        # Collect output
        stdout_content = stdout_capture.getvalue()
        stderr_content = stderr_capture.getvalue()
        
        if not success:
            if "Execution time out" in error_output:
                output = f"Execution time out after {timeout} seconds"
            else:
                output = stdout_content.rstrip(' \n')
                # remove "----" lines before Traceback
                output_lines = output.splitlines()
                for k in range(len(output_lines)-2, -1, -1):
                    line = output_lines[k]
                    if line and not line.strip('-') and "Traceback" in output_lines[k+1]:
                        output_lines[k+1] = output_lines[k+1][output_lines[k+1].find("Traceback"):]
                        output_lines.pop(k)
                        break
                    
                output = "\n".join(output_lines)
                if stderr_content:
                    output += f"\nStderr: {stderr_content}"
        else:
            output = stdout_content
            if stderr_content:
                output += f"\nWarnings: {stderr_content}"
        
        # Truncate output if too large
        if len(output) > max_output_size:
            output = output[:max_output_size] + "\n...[output truncated]..."
        output = output.replace(site_packages_dir, "/lib/python3.10/site-packages").rstrip(' \n')
        return output if output else "Script executed successfully (no output)", success

    def reset(self):
        """Optional: clear namespace if you want."""
        try:
            if hasattr(self.shell, "user_ns"):
                self.shell.user_ns.clear()
            if hasattr(self.shell, "user_global_ns"):
                self.shell.user_global_ns.clear()
            if hasattr(self.shell, "reset"):
                self.shell.reset(new_session=False)
        except Exception as e:
            print(f"Error resetting shell: {e}", file=sys.stderr)
    
    def get_namespace_info(self) -> dict:
        """Return information about the current namespace."""
        return {
            'user_variables': list(self.shell.user_ns.keys()),
            'user_globals': list(self.shell.user_global_ns.keys()) if hasattr(self.shell, 'user_global_ns') else [],
            'builtin_count': len([k for k in self.shell.user_ns.keys() if k.startswith('_')])
        }

    def get_variable_names(self) -> list:
        """Return all non-builtin variable names."""
        return [k for k in self.shell.user_ns.keys() if not k.startswith('_')]


@ray.remote
class KernelManager:
    """
    Centralized manager for all KernelActor instances.
    
    This actor lives in a single process and maintains the authoritative
    cache of request_id -> KernelActor mappings, solving the distributed
    state problem.
    """
    
    def __init__(self, max_actors: int = MAX_ACTORS):
        self.actor_cache: "OrderedDict[str, ray.actor.ActorHandle]" = OrderedDict()
        self.max_actors = max_actors
        self.manager_id = ray.get_runtime_context().get_actor_id()
        self.available_actors = []
        # NEW: Lock for thread-safe access when max_concurrency > 1
        self.cache_lock = threading.RLock()
        logger.info(f"KernelManager initialized with manager_id={self.manager_id}, max_actors={max_actors}")
    
    def get_or_create_actor(self, request_id: str) -> "ray.actor.ActorHandle":
        """
        Get existing actor or create a new one for the given request_id.
        
        This method is thread-safe within the actor (Ray actors are single-threaded).
        """
        with self.cache_lock:
            if request_id in self.actor_cache:
                actor = self.actor_cache[request_id]
                # Move to end (LRU)
                self.actor_cache.move_to_end(request_id)
                logger.debug(f"Cache HIT for request_id={request_id}")
                return actor
            
            
                                                  
            # Create new actor
            logger.info(f"Cache MISS for request_id={request_id}, creating new KernelActor")
            if self.available_actors:
                actor = self.available_actors.pop()
                future = actor.reset.remote()
                ray.get(future)
            else:
                if len(self.actor_cache) + len(self.available_actors) < self.max_actors:
                    actor = KernelActor.remote()
                else:
                    # wait for an available actor to be recycled
                    while not self.available_actors:
                        time.sleep(0.5)
                    actor = self.available_actors.pop()
                    future = actor.reset.remote()
                    ray.get(future)
            
            self.actor_cache[request_id] = actor
        
        return actor
    
    def get_stats(self) -> Dict:
        """Get statistics about the manager."""
        return {
            "active_kernels": len(self.actor_cache),
            "max_kernels": self.max_actors,
            "kernel_ids": list(self.actor_cache.keys()),
            "manager_id": self.manager_id,
        }
    
    def remove_kernel(self, request_id: str) -> bool:
        """Remove a specific kernel."""
        with self.cache_lock:
            actor = self.actor_cache.pop(request_id, None)
            if actor is not None:
                try:
                    ray.get(actor.reset.remote())
                    self.available_actors.append(actor)
                    logger.debug(f"Removed kernel for request_id={request_id}")
                    return True
                except Exception as e:
                    logger.warning(f"Error killing actor {request_id}: {e}")
                    return False
            return False
    
    def cleanup_all(self) -> None:
        """Kill all actors and clear the cache."""
        ids = list(self.actor_cache.keys())
        for rid in ids:
            self.remove_kernel(rid)
        gc.collect()
        logger.info("Cleaned up all KernelActors")


# Global reference to the manager actor (singleton)
_kernel_manager: Optional[ray.actor.ActorHandle] = None
_manager_lock = threading.Lock()


def _get_kernel_manager() -> "ray.actor.ActorHandle":
    """
    Get or create the singleton KernelManager actor.
    
    Uses a named actor so it can be shared across all Ray workers.
    """
    global _kernel_manager
    
    with _manager_lock:
        if _kernel_manager is not None:
            return _kernel_manager
        
        # Try to get existing named actor
        try:
            _kernel_manager = ray.get_actor("kernel_manager")
            logger.info("Found existing KernelManager actor")
        except ValueError:
            # Doesn't exist, create it
            logger.info("Creating new KernelManager actor")
            _kernel_manager = KernelManager.options(
                name="kernel_manager",
                lifetime="detached",
            ).remote(max_actors=MAX_ACTORS)
        
        return _kernel_manager
    
def _get_actor(request_id: str) -> "ray.actor.ActorHandle":
    """
    Get the KernelActor for the given request_id via the manager.
    """
    manager = _get_kernel_manager()
    actor = ray.get(manager.get_or_create_actor.remote(request_id))
    return actor

async def call_python_script_with_ipython_async(
    request_id: str,
    script: str,
    timeout: int = 120,
    max_output_size: int = 100 * 1024,
) -> Tuple[str, bool]:
    """
    Execute a Python script using a Ray-backed IPython actor and return (output, success).
    
    Now uses the centralized KernelManager to avoid distributed state issues.
    """
    if not isinstance(request_id, str) or not request_id.strip():
        return "Error: request_id must be a non-empty string", False

    script = script.replace("sys.stdin.buffer.read()", "sys.stdin.read()")
    if not isinstance(script, str) or not script.strip():
        return "Error: script must be a non-empty string", False

    manager = _get_kernel_manager()
    actor = ray.get(manager.get_or_create_actor.remote(request_id))
    
    loop = asyncio.get_event_loop()

    def _run():
        
        try:
            # Execute on the kernel actor with timeout
            future = actor.execute.remote(script, max_output_size, timeout)
            result = ray.get(future, timeout=timeout + 120)
            return result
        except GetTimeoutError:
            logger.warning(f"Execution timeout for request_id={request_id}")
            # Kill the timed-out actor and remove from cache
            try:
                ray.kill(actor)
            except Exception as e:
                logger.warning(f"Error killing actor on timeout: {e}")
            
            future = manager.remove_kernel.remote(request_id)
            ray.get(future)
            return f"Execution timeout after {timeout} seconds", False

    output, success = await loop.run_in_executor(None, _run)
    
    return output[:max_output_size], success


def call_python_script_with_ipython(
    request_id: str,
    script: str,
    timeout: int = 120,
    max_output_size: int = 100 * 1024,
) -> Tuple[str, bool]:
    """Synchronous wrapper around the async version."""
    output, success = asyncio.run(
        call_python_script_with_ipython_async(
            request_id, script, timeout, max_output_size
        )
    )
    return output[:max_output_size], success


def get_kernel_stats() -> Dict:
    """Get simple stats about current actor usage."""
    manager = _get_kernel_manager()
    stats = ray.get(manager.get_stats.remote())
    
    # Add process memory info
    memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
    stats["current_process_memory_mb"] = memory_mb
    
    return stats


def cleanup_all_kernels() -> None:
    """Kill all actors and clear the cache."""
    manager = _get_kernel_manager()
    ray.get(manager.cleanup_all.remote())
    gc.collect()
    logger.info("Cleaned up all KernelActors via manager")


def remove_kernel(request_id: str) -> None:
    """Remove a specific kernel by request_id."""
    manager = _get_kernel_manager()
    removed = ray.get(manager.remove_kernel.remote(request_id))
    logger.debug(f"Requested removal of KernelActor for request_id={request_id}, success={removed}")
    if removed:
        gc.collect()
        logger.debug(f"Removed KernelActor for request_id={request_id}")


# Example usage and configuration
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing with centralized KernelManager...")
    
    # Simple hello world
    result, success = call_python_script_with_ipython("test1", "print('Hello World')")
    print(f"[test1] Result: {result!r}, Success: {success}")

    # Define a variable and reuse the same request_id to check state persistence
    call_python_script_with_ipython("test1", "x = 41")
    result, success = call_python_script_with_ipython("test1", "x + 1")
    print(f"[test1] State test: {result!r}, Success: {success}")

    # Timeout test
    result, success = call_python_script_with_ipython(
        "test2",
        "a=1\nimport time\nwhile True: time.sleep(1)",
        timeout=2,
    )
    print(f"[test2] Timeout test - Result: {result!r}, Success: {success}")
    
    remove_kernel("test2")
    # Test that actor was killed
    result, success = call_python_script_with_ipython(
        "test2",
        "print(a)",
        timeout=2,
    )
    print(f"[test3] After timeout - Result: {result!r}, Success: {success}")
    
    # Test that actor was killed
    result, success = call_python_script_with_ipython(
        "test2",
        "print(b)",
        timeout=2,
    )
    print(f"[test4] After print a - Result: {result!r}, Success: {success}")

    # Stats
    stats = get_kernel_stats()
    print(f"Stats: {stats}")

    # Cleanup
    cleanup_all_kernels()