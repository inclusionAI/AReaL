# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
import time
import uuid

import pytest

from areal.elastic.control.processes import (
    OWNER_ENV,
    owned_processes,
    stop_owned_processes,
)


def test_orphaned_session_child_is_stopped_without_touching_other_owner(tmp_path):
    """Leader exit and setsid must not leave a compute process behind."""
    owner = uuid.uuid4().hex
    code = (
        "import subprocess,sys; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)'], "
        "start_new_session=True, stdout=subprocess.DEVNULL); "
        "print(p.pid, flush=True)"
    )
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    try:
        leader = subprocess.run(
            [sys.executable, "-c", code],
            env={**os.environ, OWNER_ENV: owner},
            stdout=subprocess.PIPE,
            text=True,
            check=True,
            timeout=10,
        )
        child = int(leader.stdout)
        deadline = time.monotonic() + 5
        while child not in [p.pid for p in owned_processes(owner)]:
            if time.monotonic() > deadline:
                pytest.fail("Owned child did not start")
            time.sleep(0.05)
        stop_owned_processes(owner, grace=1, force_timeout=2)
        assert not owned_processes(owner)
        assert unrelated.poll() is None
    finally:
        stop_owned_processes(owner, grace=0.1, force_timeout=2)
        unrelated.terminate()
        unrelated.wait(timeout=5)
