"""Distributed tests for Archon Engine (multi-GPU).

These tests verify Archon Engine behavior across multiple GPUs:
1. Data Parallelism (DP/FSDP) forward
2. Tensor Parallelism (TP) forward
3. Context Parallelism (CP/Ulysses) forward

Run tests:
    pytest areal/tests/experimental/archon/test_distributed.py -v -m multi_gpu

Note: These tests require multiple GPUs and are marked with @pytest.mark.multi_gpu.
"""

import subprocess

import pytest
import torch

from areal.platforms import current_platform
from areal.tests.experimental.archon.utils import run_torchrun_test
from areal.utils.network import find_free_ports

# Skip if no CUDA available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


# =============================================================================
# Helper Functions
# =============================================================================


def _run_tp_test_with_torchrun(n_gpus: int, tp_size: int):
    """Run FSDP+TP Archon forward test."""
    port = find_free_ports(1)[0]
    try:
        subprocess.run(
            [
                "torchrun",
                f"--nproc_per_node={n_gpus}",
                "--nnodes=1",
                "--master-addr=localhost",
                f"--master_port={port}",
                "areal/tests/experimental/archon/torchrun/run_tp_forward.py",
                f"--tp_size={tp_size}",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Test failed with error: {e.stderr}")


def _run_cp_test_with_torchrun(n_gpus: int, cp_size: int):
    """Run FSDP+CP (Ulysses) Archon forward test."""
    port = find_free_ports(1)[0]
    try:
        subprocess.run(
            [
                "torchrun",
                f"--nproc_per_node={n_gpus}",
                "--nnodes=1",
                "--master-addr=localhost",
                f"--master_port={port}",
                "areal/tests/experimental/archon/torchrun/run_cp_forward.py",
                f"--cp_size={cp_size}",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Test failed with error: {e.stderr}")


# =============================================================================
# Data Parallelism (FSDP) Tests
# =============================================================================


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_archon_forward_dp_2gpu():
    """Test Archon Engine forward_batch with Data Parallelism (2 GPUs)."""
    if current_platform.device_count() < 2:
        pytest.skip("This test requires 2 GPUs")
    run_torchrun_test(
        "areal/tests/experimental/archon/torchrun/run_forward.py",
        n_gpus=2,
    )


# =============================================================================
# Tensor Parallelism (TP) Tests
# =============================================================================


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_archon_tp_forward_2gpu():
    """Test Archon Engine forward with FSDP+TP on 2 GPUs (dp=1, tp=2)."""
    if current_platform.device_count() < 2:
        pytest.skip("This test requires 2 GPUs")
    _run_tp_test_with_torchrun(2, 2)


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_archon_tp_forward_4gpu():
    """Test Archon Engine forward with FSDP+TP on 4 GPUs (dp=2, tp=2)."""
    if current_platform.device_count() < 4:
        pytest.skip("This test requires 4 GPUs")
    _run_tp_test_with_torchrun(4, 2)


# =============================================================================
# Context Parallelism (CP/Ulysses) Tests
# =============================================================================


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_archon_cp_forward_2gpu():
    """Test Archon Engine forward with CP (Ulysses) on 2 GPUs (dp=1, cp=2)."""
    if current_platform.device_count() < 2:
        pytest.skip("This test requires 2 GPUs")
    _run_cp_test_with_torchrun(2, 2)


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_archon_cp_forward_4gpu():
    """Test Archon Engine forward with CP (Ulysses) on 4 GPUs (dp=2, cp=2)."""
    if current_platform.device_count() < 4:
        pytest.skip("This test requires 4 GPUs")
    _run_cp_test_with_torchrun(4, 2)


# =============================================================================
# Expert Parallelism (EP) Tests
# =============================================================================


def _run_ep_test_with_torchrun(n_gpus: int, test_type: str, output: str):
    """Run EP test with torchrun.

    Args:
        n_gpus: Number of GPUs to use
        test_type: Type of EP test (ep_tp_forward, ep_only_forward, etc.)
        output: Output file path for test result verification
    """
    port = find_free_ports(1)[0]
    try:
        subprocess.run(
            [
                "torchrun",
                f"--nproc_per_node={n_gpus}",
                "--nnodes=1",
                "--master-addr=localhost",
                f"--master_port={port}",
                "areal/tests/experimental/archon/torchrun/run_ep_tests.py",
                f"--test_type={test_type}",
                f"--output={output}",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Test failed with error: {e.stderr}")

    # Verify test result from output file
    with open(output) as f:
        result = f.read().strip()
    assert result == "Passed", f"Test failed: {result}"


# --- EP + TP Tests (ep=world_size, tp=world_size) ---


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_archon_ep_tp_forward_2gpu(tmp_path_factory):
    """Test EP+TP forward numerical correctness (ep=2, tp=2) on 2 GPUs.

    Verify: EP model output matches non-EP golden model output.
    """
    if current_platform.device_count() < 2:
        pytest.skip("This test requires 2 GPUs")
    output = tmp_path_factory.mktemp("test_output") / "ep_tp_forward.out"
    _run_ep_test_with_torchrun(2, "ep_tp_forward", str(output))


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_archon_ep_tp_weight_sync_2gpu(tmp_path_factory):
    """Test EP+TP weight gather and roundtrip (ep=2, tp=2) on 2 GPUs.

    Verify: Gathered weights match original, cross-rank consistency, roundtrip.
    """
    if current_platform.device_count() < 2:
        pytest.skip("This test requires 2 GPUs")
    output = tmp_path_factory.mktemp("test_output") / "ep_tp_weight_sync.out"
    _run_ep_test_with_torchrun(2, "ep_tp_weight_sync", str(output))


# --- EP Only Tests (ep=world_size, tp=1, cp=1) ---


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_archon_ep_only_forward_2gpu(tmp_path_factory):
    """Test EP only forward (ep=2, tp=1, cp=1) on 2 GPUs.

    Verify: EP model output matches non-EP golden model output.
    """
    if current_platform.device_count() < 2:
        pytest.skip("This test requires 2 GPUs")
    output = tmp_path_factory.mktemp("test_output") / "ep_only_forward.out"
    _run_ep_test_with_torchrun(2, "ep_only_forward", str(output))


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_archon_ep_only_weight_sync_2gpu(tmp_path_factory):
    """Test EP only weight sync (ep=2, tp=1, cp=1) on 2 GPUs.

    Verify: Gathered weights match original, cross-rank consistency, roundtrip.
    """
    if current_platform.device_count() < 2:
        pytest.skip("This test requires 2 GPUs")
    output = tmp_path_factory.mktemp("test_output") / "ep_only_weight_sync.out"
    _run_ep_test_with_torchrun(2, "ep_only_weight_sync", str(output))


# --- EP + CP Tests (ep=2, tp=1, cp=2, 4 GPU) ---


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_archon_ep_cp_forward_4gpu(tmp_path_factory):
    """Test EP+CP forward (ep=2, tp=1, cp=2) on 4 GPUs.

    Verify: EP+CP model output matches non-EP golden model output.
    """
    if current_platform.device_count() < 4:
        pytest.skip("This test requires 4 GPUs")
    output = tmp_path_factory.mktemp("test_output") / "ep_cp_forward.out"
    _run_ep_test_with_torchrun(4, "ep_cp_forward", str(output))


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_archon_ep_cp_weight_sync_4gpu(tmp_path_factory):
    """Test EP+CP weight sync (ep=2, tp=1, cp=2) on 4 GPUs.

    Verify: Gathered weights match original, cross-rank consistency, roundtrip.
    """
    if current_platform.device_count() < 4:
        pytest.skip("This test requires 4 GPUs")
    output = tmp_path_factory.mktemp("test_output") / "ep_cp_weight_sync.out"
    _run_ep_test_with_torchrun(4, "ep_cp_weight_sync", str(output))


# --- EP State Dict Update Tests ---


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_archon_ep_state_dict_update_2gpu(tmp_path_factory):
    """Test state dict correctness after optimizer step (ep=2, tp=2) on 2 GPUs.

    Verify:
    1. Forward + backward + optimizer.step()
    2. Gather EP weights to full state_dict
    3. Load to non-EP model, forward outputs should match.
    """
    if current_platform.device_count() < 2:
        pytest.skip("This test requires 2 GPUs")
    output = tmp_path_factory.mktemp("test_output") / "state_dict_update.out"
    _run_ep_test_with_torchrun(2, "state_dict_update", str(output))
