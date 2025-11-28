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
                # Parent is already gone, but we still need to wait for children.
                pass

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
