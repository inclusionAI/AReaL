"""Synthetic CPU job for restart plumbing; never a GPU/AReaL training benchmark.

Uses the actual AReaL snapshot core without importing its GPU dependencies.
The toy trainer mirrors the bridge's stop/restore interfaces; tests of native
PPOTrainer and backend state remain in AReaL/tests/test_reconfiguration*.py.
"""

import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from autopilot_harness.execution.protocol import LAUNCH_ENV, read_json
from autopilot_harness.execution.runtime import run_training

CONFIG = {"steps": 8, "delay": 0.08}
ROOT = Path(__file__).resolve().parents[3]
CORE = ROOT / "areal/infra/reconfiguration.py"
spec = importlib.util.spec_from_file_location("fixture_snapshot_core", CORE)
M = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = M
spec.loader.exec_module(M)
FAULT = os.environ.get("AUTOPILOT_FIXTURE_FAULT", "")
launch = read_json(Path(os.environ[LAUNCH_ENV]))
run_root = Path(os.environ[LAUNCH_ENV]).parents[2]


@dataclass
class ToyRestore:
    released: bool = False

    def resume(self, trainer):
        self.released = True


class ToyTrainer:
    def __init__(self, restore):
        self.request = restore
        self.count = 0
        self._snapshot_restore = ToyRestore() if restore else None
        self.snapshot_restore_result = None
        self.worker = None
        self.lease = run_root / "fixture-resource.lock"

    def __enter__(self):
        self.lease.touch(exist_ok=False)
        self.worker = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"]
        )
        return self

    def __exit__(self, *args):
        self.worker.terminate()
        self.worker.wait(timeout=5)
        self.lease.unlink()
        if FAULT == "cleanup" and self.request is None:
            raise RuntimeError("fixture cleanup failure")

    def state_dict(self):
        return {"count": self.count}

    def load_state_dict(self, state):
        self.count = state["count"]

    def checkpoint(self):
        def save(path):
            (path / "model.json").write_text(json.dumps({"weight": self.count}))

        def load(path):
            self.count = json.loads((path / "model.json").read_text())["weight"]

        return M.CheckpointParticipant("synthetic-counter-v1", save, lambda: None, load)

    def train(self, *, reconfiguration):
        reconfiguration.bind(
            checkpoints={"actor": self.checkpoint()}, statefuls={"progress": self}
        )
        try:
            if self.request:
                if FAULT == "restore":
                    raise RuntimeError("fixture restore failure")
                manager = M.ReconfigurationManager(
                    checkpoints={"actor": self.checkpoint()},
                    statefuls={"progress": self},
                )
                self.snapshot_restore_result = manager.restore_snapshot(
                    self.request.snapshot,
                    expected_context=self.request.expected_context,
                )
                if self.count < CONFIG["steps"]:
                    self._snapshot_restore.resume(self)
            limit = 1000 if FAULT == "no_snapshot" else CONFIG["steps"]
            for step in range(self.count, limit):
                time.sleep(CONFIG["delay"])
                self.count += 1
                if FAULT == "no_snapshot":
                    continue
                if FAULT == "last_step" and self.count != limit:
                    continue
                info = dict(
                    epoch=0,
                    epoch_step=step,
                    global_step=step,
                    steps_per_epoch=CONFIG["steps"],
                )
                reconfiguration.begin_step_snapshot(step_info=info)
                result = reconfiguration.finish_step_snapshot(policy_version=self.count)
                if result:
                    if FAULT == "corrupt":
                        (Path(result.snapshot.path) / "manifest.json").write_text("{}")
                    return result
            (Path(os.environ[LAUNCH_ENV]).parent / "fixture-final.json").write_text(
                json.dumps({"count": self.count, "restored": self.request is not None})
            )
        finally:
            reconfiguration.close()


if __name__ == "__main__":

    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    identity = hashlib.sha256(
        CORE.read_bytes() + Path(__file__).read_bytes()
    ).hexdigest()
    if FAULT == "source_drift" and launch["restore"]:
        identity = "different-fixture-runtime"
    run_training(
        ToyTrainer,
        configuration=CONFIG,
        runtime_source_id="fixture:" + identity,
        snapshot_api=M,
    )
