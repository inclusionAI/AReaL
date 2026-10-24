# SPDX-License-Identifier: Apache-2.0
"""A task-long offline bubblewrap service with fresh Python per tool call."""

from __future__ import annotations

import inspect
import json
import os
import selectors
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

from examples.task_workflows.environments.process import run_limited
from examples.task_workflows.environments.python import PythonSandbox, ToolSettings

_SETUP = r"""
import ctypes
libc=ctypes.CDLL(None,use_errno=True)
for path,size in [(b"/work",64*1024**2),(b"/tmp",32*1024**2)]:
    if libc.mount(None,path,None,32,("size="+str(size)).encode()) != 0:
        raise OSError(ctypes.get_errno(),"Cannot bound math tool tmpfs")
# The legacy bubblewrap requires namespace uid 0 for mount setup. Remove every
# capability (including the bounding set) and forbid privilege gain before any
# generated code or requests are read. This uid maps only to the sandbox user.
for capability in range(64):
    libc.prctl(24,capability,0,0,0)  # PR_CAPBSET_DROP; unknown capability => EINVAL
class Header(ctypes.Structure):
    _fields_=[("version",ctypes.c_uint32),("pid",ctypes.c_int)]
class Data(ctypes.Structure):
    _fields_=[("effective",ctypes.c_uint32),("permitted",ctypes.c_uint32),("inheritable",ctypes.c_uint32)]
header=Header(0x20080522,0)
capabilities=(Data*2)()
if libc.capset(ctypes.byref(header),capabilities) != 0 or libc.prctl(38,1,0,0,0) != 0:
    raise OSError(ctypes.get_errno(),"Cannot drop math tool privileges")
if libc.capget(ctypes.byref(header),capabilities) != 0:
    raise OSError(ctypes.get_errno(),"Cannot verify math tool capabilities")
remaining=[cap for cap in range(64) if libc.prctl(23,cap,0,0,0)>0]
if remaining or any(row.effective or row.permitted or row.inheritable for row in capabilities):
    raise RuntimeError("Math tool retained capabilities")
"""


class ArenaPython(PythonSandbox):
    """Reuse the tool contract while keeping its work and tmp files size bounded."""

    def __init__(self, settings: ToolSettings, task_id: str) -> None:
        super().__init__(settings, task_id, image_id="arena-bubblewrap")
        if sys.platform != "linux" or shutil.which("bwrap") is None:
            raise RuntimeError("The Arena math tool requires Linux and bubblewrap")
        self._bwrap = shutil.which("bwrap")
        self._service: subprocess.Popen | None = None
        self._probe: dict = {}

    def _base_command(self) -> list[str]:
        command = [
            self._bwrap,
            "--die-with-parent",
            "--new-session",
            "--unshare-all",
            "--uid",
            "0",
            "--gid",
            "0",
            "--cap-drop",
            "ALL",
            "--cap-add",
            "CAP_SYS_ADMIN",
            "--cap-add",
            "CAP_SETPCAP",
            "--clearenv",
            "--tmpfs",
            "/tmp",
        ]
        # Never expose /envarena, /proc, HOME, credentials or agent artifacts.
        roots = {
            Path("/usr"),
            Path("/lib"),
            Path("/lib64"),
            Path(sys.base_prefix),
            Path(sys.prefix),
        }
        for root in sorted(roots):
            if root.exists():
                command += ["--ro-bind", str(root), str(root)]
        return command + [
            "--dev",
            "/dev",
            "--tmpfs",
            "/work",
            "--chdir",
            "/work",
            "--setenv",
            "PATH",
            os.defpath,
            "--setenv",
            "PYTHONDONTWRITEBYTECODE",
            "1",
        ]

    def _receive(self) -> dict:
        if self._service is None or self._service.stdout is None:
            raise RuntimeError("No math tool service")
        with selectors.DefaultSelector() as selector:
            selector.register(self._service.stdout, selectors.EVENT_READ)
            if not selector.select(self.settings.timeout + 10):
                raise subprocess.TimeoutExpired(
                    "math tool service", self.settings.timeout + 10
                )
        line = self._service.stdout.readline(131072)
        if not line:
            error = self._service.stderr.read(4096).decode(errors="replace")
            raise subprocess.SubprocessError("Math tool service exited: " + error)
        return json.loads(line)

    def _start(self) -> None:
        if self._service is not None:
            return
        # The service contains no model credentials. Only this outer process
        # talks to Hermes; generated code runs in a new interpreter below.
        service = (
            "import json,os,selectors,signal,socket,subprocess,sys,time\n"
            + inspect.getsource(run_limited)
            + _SETUP
            + "\nprint(json.dumps({'ready':True,'proc_visible':os.path.exists('/proc'),"
            "'arena_visible':os.path.exists('/envarena'),'environment':sorted(os.environ),"
            "'work_bytes':os.statvfs('/work').f_frsize*os.statvfs('/work').f_blocks,"
            "'tmp_bytes':os.statvfs('/tmp').f_frsize*os.statvfs('/tmp').f_blocks,"
            "'capabilities_remaining':remaining,'no_new_privs':libc.prctl(39,0,0,0,0),"
            "'network_interfaces':socket.if_nameindex()}),flush=True)\n"
            "for line in sys.stdin:\n"
            " request=json.loads(line)\n"
            " result=run_limited(request['command'],timeout=request['timeout'],"
            "output_bytes=request['output_bytes'],process_group=True,separate_stderr=True)\n"
            " print(json.dumps(result),flush=True)\n"
        )
        self._service = subprocess.Popen(
            self._base_command() + [sys.executable, "-I", "-u", "-c", service],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={"PATH": os.defpath},
            start_new_session=True,
        )
        try:
            self._probe = self._receive()
            if (
                not self._probe.get("ready")
                or self._probe["proc_visible"]
                or self._probe["arena_visible"]
            ):
                raise RuntimeError("The math tool isolation probe did not pass")
        except BaseException:
            self.close()
            raise
        self.container = "bubblewrap:" + self.task_id

    def _run_code(self, code: str) -> dict[str, Any]:
        if self._service is None or self._service.stdin is None:
            raise RuntimeError("No math tool service")
        limited = (
            "import os,resource\n"
            f"resource.setrlimit(resource.RLIMIT_AS,({self.settings.memory_mb}*1048576,)*2)\n"
            f"resource.setrlimit(resource.RLIMIT_CPU,({max(1, int(self.settings.timeout))},)*2)\n"
            "resource.setrlimit(resource.RLIMIT_FSIZE,(64*1048576,)*2)\n"
            "resource.setrlimit(resource.RLIMIT_NOFILE,(64,)*2)\n"
            "resource.setrlimit(resource.RLIMIT_NPROC,(64,)*2)\n"
            "os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})\n"
            "fd=os.open(os.devnull,os.O_RDONLY);os.dup2(fd,0);os.close(fd)\n"
            f"exec(compile({code!r},'<math_python>','exec'))\n"
        )
        request = {
            "command": [sys.executable, "-I", "-u", "-c", limited],
            "timeout": self.settings.timeout,
            "output_bytes": self.settings.max_output_bytes,
        }
        self._service.stdin.write((json.dumps(request) + "\n").encode())
        self._service.stdin.flush()
        return self._receive()

    def close(self) -> None:
        process, self._service = self._service, None
        self.container = None
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            if process.poll() is None:
                process.kill()
            process.wait(timeout=10)
            for stream in [process.stdin, process.stdout, process.stderr]:
                if stream is not None:
                    stream.close()

    def metadata(self) -> dict[str, Any]:
        return {
            **super().metadata(),
            "backend": "arena-bubblewrap",
            "isolation_probe": self._probe,
            "limits": {
                "work_tmpfs_bytes": 64 * 1024**2,
                "tmp_tmpfs_bytes": 32 * 1024**2,
                "memory_mb": self.settings.memory_mb,
                "cpu_affinity": 1,
                "pids_rlimit": 64,
            },
        }
