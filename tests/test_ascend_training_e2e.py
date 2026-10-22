# SPDX-License-Identifier: Apache-2.0

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

import psutil
import pytest

pytestmark = [pytest.mark.multi_npu, pytest.mark.slow, pytest.mark.ci]


def test_geometry3k_two_steps_evaluates_and_saves(tmp_path: Path) -> None:
    """Exercise PP2/TP2 training, vLLM graph/MTP rollout, evaluation and HF save."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available() or torch.npu.device_count() < 8:
        pytest.skip("Requires eight Ascend NPUs")
    model_path = os.environ.get("AREAL_TEST_QWEN35_PATH")
    assert model_path and (Path(model_path) / "config.json").is_file(), (
        "Set AREAL_TEST_QWEN35_PATH to a cached Qwen3.5-0.8B model"
    )
    output = Path(os.environ.get("AREAL_E2E_OUTPUT_DIR", str(tmp_path / "output")))
    output.mkdir(parents=True, exist_ok=True)
    work_parent = os.environ.get("AREAL_E2E_WORK_DIR")
    if work_parent:
        Path(work_parent).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="geometry3k-e2e-", dir=work_parent
    ) as workdir:
        root = Path(workdir)
        cmd = [
            sys.executable,
            "-m",
            "tests.run_ascend_training_e2e",
            "--config",
            "tests/configs/ascend_geometry3k_grpo.yaml",
            f"actor.path={model_path}",
            f"cluster.fileroot={root}",
            f"cluster.name_resolve.nfs_record_root={root / 'name_resolve'}",
            f"trial_name=ci-{uuid.uuid4().hex[:12]}",
        ]
        with (output / "controller.log").open("w") as log:
            process = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            # Let CI timeout/cancellation unwind the process and directory cleanup.
            previous_sigterm = signal.signal(signal.SIGTERM, signal.default_int_handler)
            try:
                returncode = process.wait(timeout=900)
                assert returncode == 0, (
                    f"Training exited with {returncode}; see {output / 'controller.log'}\n"
                    + "".join(
                        (output / "controller.log")
                        .read_text()
                        .splitlines(keepends=True)[-120:]
                    )
                )
                evidence = json.loads((root / "completion.json").read_text())
                assert evidence["completed_train_steps"] == 2
                assert evidence["completed_evaluations"] == 1
                shutil.copy2(root / "completion.json", output / "completion.json")
            except subprocess.TimeoutExpired:
                pytest.fail(
                    "Training timed out. Controller log tail:\n"
                    + "".join(
                        (output / "controller.log")
                        .read_text()
                        .splitlines(keepends=True)[-120:]
                    )
                )
            finally:
                signal.signal(signal.SIGTERM, previous_sigterm)
                if process.poll() is None:
                    parent = psutil.Process(process.pid)
                    children = parent.children(recursive=True)
                    for child in reversed(children):
                        try:
                            child.kill()
                        except psutil.NoSuchProcess:
                            pass
                    process.kill()
                    process.wait(timeout=30)
                    psutil.wait_procs(children, timeout=10)
                if (root / "logs").exists():
                    shutil.copytree(
                        root / "logs", output / "workers", dirs_exist_ok=True
                    )
    assert not root.exists(), f"Temporary training files remain at {root}"
