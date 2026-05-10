"""End-to-end test for LoRA disk-based weight synchronization.

This test launches ``torchrun`` with the helper script
``tests/torchrun/run_lora_disk_sync.py`` which:

1. Creates a small Qwen3-0.6B model with FSDP + LoRA + ``weight_update_mode="disk"``.
2. Writes a PEFT-format adapter checkpoint via
   ``FSDPEngine._save_model_to_hf`` and verifies the on-disk artefacts.
3. Verifies the model can still run a forward pass after the save.

The test requires GPUs and is marked ``@pytest.mark.slow`` and
``@pytest.mark.sglang`` following AReaL test conventions.

Usage:
  pytest tests/test_lora_disk_sync_e2e.py -v
"""

import os
import subprocess

import pytest

from areal.api.alloc_mode import ModelAllocation
from areal.infra.platforms import current_platform
from areal.utils.network import find_free_ports

MODEL_PATH = "/workspace/models/Qwen3-0.6B"


def _run_torchrun_test(alloc_mode: str, output: str, n_gpus: int | None = None):
    """Launch the torchrun helper script and check the result.

    Parameters
    ----------
    alloc_mode : str
        Backend allocation string, e.g. ``"fsdp:d1t1"``.
    output : str
        Path to the result file that the torchrun script writes.
    n_gpus : int, optional
        Override number of GPUs.  If ``None``, it is derived from
        ``alloc_mode``.
    """
    port = find_free_ports(1)[0]
    if n_gpus is None:
        n_gpus = ModelAllocation.from_str(alloc_mode).parallel.world_size

    cmd = [
        "torchrun",
        f"--nproc_per_node={n_gpus}",
        "--nnodes=1",
        "--master-addr=localhost",
        f"--master_port={port}",
        "tests/torchrun/run_lora_disk_sync.py",
        f"--backend={alloc_mode}",
        f"--output={output}",
        f"--model-path={MODEL_PATH}",
    ]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    try:
        subprocess.run(
            cmd,
            check=True,
            timeout=300,
            env=env,
        )
    except subprocess.CalledProcessError as e:
        pytest.fail(
            f"torchrun failed with exit code {e.returncode}.\n"
            "See the live torchrun output above for details."
        )
    except subprocess.TimeoutExpired:
        pytest.fail("torchrun timed out after 300 seconds.")

    with open(output) as f:
        result_text = f.read().strip()
    assert result_text == "Passed", f"Test failed: {result_text}"


# ---------------------------------------------------------------------------
# Single-GPU test (dp=1, tp=1)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.sglang
def test_lora_disk_sync_single_gpu(tmp_path_factory):
    """Test LoRA disk sync with 1 GPU using FSDP engine."""
    if current_platform.device_count() < 1:
        pytest.skip("Test requires at least 1 GPU")

    output = tmp_path_factory.mktemp("test_output") / "lora_disk_sync_1gpu.out"
    _run_torchrun_test("fsdp:d1t1", str(output), n_gpus=1)


# ---------------------------------------------------------------------------
# Multi-GPU test (dp=2, tp=1)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.sglang
def test_lora_disk_sync_multi_gpu_dp2(tmp_path_factory):
    """Test LoRA disk sync with 2 GPUs (data parallel = 2)."""
    if current_platform.device_count() < 2:
        pytest.skip("Test requires at least 2 GPUs")

    output = tmp_path_factory.mktemp("test_output") / "lora_disk_sync_dp2.out"
    _run_torchrun_test("fsdp:d2t1", str(output), n_gpus=2)


# ---------------------------------------------------------------------------
# Multi-GPU test (dp=4, tp=1)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.sglang
def test_lora_disk_sync_multi_gpu_dp4(tmp_path_factory):
    """Test LoRA disk sync with 4 GPUs (data parallel = 4)."""
    if current_platform.device_count() < 4:
        pytest.skip("Test requires at least 4 GPUs")

    output = tmp_path_factory.mktemp("test_output") / "lora_disk_sync_dp4.out"
    _run_torchrun_test("fsdp:d4t1", str(output), n_gpus=4)


# ---------------------------------------------------------------------------
# Multi-GPU test (dp=2, tp=2) -- 4 GPUs with tensor parallel
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.sglang
def test_lora_disk_sync_multi_gpu_dp2_tp2(tmp_path_factory):
    """Test LoRA disk sync with 4 GPUs (dp=2, tp=2)."""
    if current_platform.device_count() < 4:
        pytest.skip("Test requires at least 4 GPUs")

    output = tmp_path_factory.mktemp("test_output") / "lora_disk_sync_dp2_tp2.out"
    _run_torchrun_test("fsdp:d2t2", str(output), n_gpus=4)
