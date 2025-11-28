import os
import signal
import sys
import threading

import psutil

from areal.utils import logging

logger = logging.getLogger(__name__)


def kill_process_tree(
    parent_pid: int | None = None,
    timeout: int = 5,
    include_parent: bool = True,
    skip_pid: int | None = None,
    graceful: bool = True,
) -> None:
    """Terminate a process and all its children recursively.

    By default, this function performs graceful termination by sending SIGTERM first,
    waiting for the specified timeout, then force-killing any remaining processes with SIGKILL.
    For immediate termination, set graceful=False to skip the graceful shutdown attempt.

    Args:
        parent_pid: Process ID to terminate. If None, defaults to current process (os.getpid())
                   and sets include_parent=False.
        timeout: Seconds to wait for graceful termination before force-killing (only used when graceful=True).
                Default is 5 seconds.
        include_parent: Whether to terminate the parent process itself. If False, only children are terminated.
                       Default is True.
        skip_pid: Optional child PID to skip during termination. This child will not be sent any signals.
                 Default is None (no children skipped).
        graceful: If True (default), attempts graceful shutdown with SIGTERM before SIGKILL.
                 If False, immediately sends SIGKILL to all processes.

    Notes:
        - When killing the current process (parent_pid == os.getpid()), the function will
          call sys.exit(0) after sending the kill signal in aggressive mode.
        - For processes that cannot be killed with SIGKILL (e.g., PID=1 in Kubernetes),
          a SIGQUIT signal is also sent as a fallback in aggressive mode.
        - The SIGCHLD signal handler is reset to avoid spammy logs (only in main thread).
        - All psutil.NoSuchProcess exceptions are caught and ignored (process already terminated).

    Examples:
        # Graceful termination with default 5s timeout
        kill_process_tree(1234)

        # Graceful termination with custom 10s timeout
        kill_process_tree(1234, timeout=10)

        # Immediate force-kill
        kill_process_tree(1234, graceful=False)

        # Kill only children, not parent
        kill_process_tree(1234, include_parent=False)

        # Skip specific child process
        kill_process_tree(1234, skip_pid=5678)

        # Graceful shutdown of current process
        kill_process_tree(None, graceful=True)
    """
    # Remove sigchld handler to avoid spammy logs (only in main thread)
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)

    # Handle None parent_pid - defaults to current process
    if parent_pid is None:
        parent_pid = os.getpid()
        include_parent = False

    # Get process tree
    try:
        parent = psutil.Process(parent_pid)
        children = parent.children(recursive=True)
    except psutil.NoSuchProcess:
        logger.info(f"Process {parent_pid} already terminated")
        return

    # Filter skip_pid from children
    if skip_pid is not None:
        children = [c for c in children if c.pid != skip_pid]

    # Terminate based on mode
    if graceful:
        # Graceful mode: SIGTERM → wait → SIGKILL
        logger.info(
            f"Sending SIGTERM to process {parent_pid} and {len(children)} children"
        )

        # Send SIGTERM to all children
        for child in children:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass

        # Send SIGTERM to parent if requested
        if include_parent:
            try:
                parent.terminate()
            except psutil.NoSuchProcess:
                logger.info(f"Process {parent_pid} already terminated")
                return

        # Wait for graceful shutdown
        procs_to_wait = children + ([parent] if include_parent else [])
        gone, alive = psutil.wait_procs(procs_to_wait, timeout=timeout)

        # Force kill any remaining processes
        if alive:
            logger.warning(
                f"Force killing {len(alive)} processes that didn't terminate gracefully"
            )
            for proc in alive:
                try:
                    proc.kill()
                except psutil.NoSuchProcess:
                    pass

            # Final wait to ensure they're gone
            psutil.wait_procs(alive, timeout=1)

        logger.info(f"Successfully cleaned up process tree for PID {parent_pid}")
    else:
        # Aggressive mode: immediate SIGKILL
        for child in children:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass

        if include_parent:
            try:
                if parent_pid == os.getpid():
                    parent.kill()
                    sys.exit(0)

                parent.kill()

                # Sometime processes cannot be killed with SIGKILL (e.g, PID=1 launched by kubernetes),
                # so we send an additional signal to kill them.
                parent.send_signal(signal.SIGQUIT)
            except psutil.NoSuchProcess:
                pass
