"""Distributed tests for Archon Engine (multi-GPU).

These tests verify Archon Engine behavior across multiple GPUs:
1. Data Parallelism (DP/FSDP) forward
2. Tensor Parallelism (TP) forward
3. Context Parallelism (CP/Ulysses) forward
4. TP + AC + compile compatibility

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
# TP + AC + compile Compatibility Tests
# =============================================================================


def _run_tp_ac_compile_test_with_torchrun(n_gpus: int, tp_size: int):
    """Run TP + AC + compile compatibility test."""
    port = find_free_ports(1)[0]
    try:
        subprocess.run(
            [
                "torchrun",
                f"--nproc_per_node={n_gpus}",
                "--nnodes=1",
                "--master-addr=localhost",
                f"--master_port={port}",
                "areal/tests/experimental/archon/torchrun/run_tp_ac_compile.py",
                f"--tp_size={tp_size}",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Test failed with error: {e.stderr}")


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_tp_ac_compile_compatibility_2gpu():
    """Test TP + AC + compile doesn't trigger dynamo recompilation (2 GPUs).

    This test verifies that the _WaitAsyncWrapper fix allows the combination
    of Tensor Parallelism, Activation Checkpointing, and torch.compile to work
    together without dynamo recompilation warnings.
    """
    if current_platform.device_count() < 2:
        pytest.skip("This test requires 2 GPUs")
    _run_tp_ac_compile_test_with_torchrun(2, 2)
