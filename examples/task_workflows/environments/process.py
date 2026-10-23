# SPDX-License-Identifier: Apache-2.0
"""Bounded subprocess execution shared with Python 3.6.8 compute hosts."""

import os
import selectors
import signal
import subprocess
import time


def run_limited(
    command, *, timeout, output_bytes, truncate=False, process_group=False, env=None
):
    """Bound a subprocess's combined output and runtime without buffering it all."""
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=process_group,
        env=env,
    )
    output = bytearray()
    status = "completed"
    truncated = False
    assert process.stdout is not None
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while selector.get_map():
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    status = "timeout"
                    break
                for key, _ in selector.select(min(remaining, 0.2)):
                    chunk = os.read(key.fileobj.fileno(), 4096)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    available = output_bytes - len(output)
                    output.extend(chunk[:available])
                    if len(chunk) > available:
                        truncated = True
                        if not truncate:
                            status = "output_limit"
                            break
                if status != "completed":
                    break
        if status == "completed":
            try:
                process.wait(timeout=max(0.01, timeout - (time.monotonic() - started)))
            except subprocess.TimeoutExpired:
                status = "timeout"
    finally:
        if process_group:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if process.poll() is None:
            process.kill()
        process.wait()
        process.stdout.close()
    return {
        "status": "output_limit" if truncated and status == "completed" else status,
        "exit_code": process.returncode,
        "output": output.decode(errors="replace"),
        "seconds": time.monotonic() - started,
    }
