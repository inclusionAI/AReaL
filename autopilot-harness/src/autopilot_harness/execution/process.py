"""Owned POSIX process groups for standard, non-detaching local jobs.

Remote schedulers and commands that escape the group require another backend.
No PID from a file or an unrelated process can be adopted by this backend.
"""

import errno
import os
import signal
import subprocess
import time
from pathlib import Path


class LocalProcess:
    def __init__(self, argv: tuple[str, ...], cwd: Path, environment: dict, log: Path):
        if os.name != "posix":
            raise NotImplementedError("local restart requires POSIX process groups")
        with log.open("xb") as output:
            os.chmod(log, 0o600)
            self.process = subprocess.Popen(
                argv,
                cwd=cwd,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        self.pgid = self.process.pid
        self._gone = False

    def poll(self) -> int | None:
        return self.process.poll()

    def group_gone(self) -> bool:
        if self._gone:
            return True
        if self.poll() is None:
            return False
        try:
            os.killpg(self.pgid, 0)
        except OSError as error:
            if error.errno != errno.ESRCH:
                raise  # EPERM is not evidence of exit.
            self._gone = True
        return self._gone

    def terminate(self, timeout: float = 5) -> None:
        """Explicit cancellation/cleanup of this owned group, never an exit proof."""
        # A completion/error receipt can precede normal interpreter exit by a
        # few milliseconds. Reap that exit before sending cancellation signals.
        deadline = time.monotonic() + min(timeout, 0.1)
        while time.monotonic() < deadline:
            if self.group_gone():
                return
            time.sleep(0.01)
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(self.pgid, sig)
            except ProcessLookupError:
                pass
            except PermissionError:
                # An exit may race with the signal. Only ESRCH from a fresh
                # group probe permits success; permission denial itself does not.
                if self.group_gone():
                    return
                raise
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if self.group_gone():
                    return
                time.sleep(0.02)
        raise TimeoutError("owned runtime process group did not exit")
