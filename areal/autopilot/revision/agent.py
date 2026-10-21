# SPDX-License-Identifier: Apache-2.0

"""Agent protocol and a provider-independent JSON command adapter."""

import json
import math
import os
import signal
import subprocess
from dataclasses import dataclass
from typing import Protocol

from areal.autopilot.revision.contracts import Proposal, ProposalRequest


class RevisionAgent(Protocol):
    def propose(self, request: ProposalRequest) -> Proposal: ...


@dataclass(frozen=True)
class CommandRevisionAgent:
    """Call a configured agent: request JSON on stdin, proposal JSON on stdout.

    The command can wrap any model/provider. It receives calibration and aggregate
    feedback, never the evaluator's held-out labels or reference simulator object.
    """

    command: tuple[str, ...]
    timeout_seconds: float = 120

    def propose(self, request: ProposalRequest) -> Proposal:
        if (
            not self.command
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("Agent requires a command and positive timeout")
        process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        )
        try:
            output, error = process.communicate(
                json.dumps(request.to_dict(), allow_nan=False),
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            raise TimeoutError("Revision agent timed out") from None
        if process.returncode:
            raise RuntimeError(
                f"Revision agent exited {process.returncode}: {error[-2000:]}"
            )
        if len(output.encode()) > 256_000:
            raise ValueError("Agent response exceeds 256 KB")
        return Proposal(**json.loads(output))
