"""FSDP Pipeline Parallelism (PP) distributed tests.

These tests require multiple GPUs and use torchrun for distributed execution.
They validate the FSDP-specific pipeline parallelism implementation that works
with HuggingFace-style models (nn.ModuleList for layers).

Run tests:
    pytest tests/fsdp/test_fsdp_distributed_pp.py -v -m multi_gpu

Test configuration:
    2 GPU Tests (Core PP):
        - test_fsdp_pp_forward_2gpu: PP=2 forward pass with FSDP PP split
        - test_fsdp_pp_backward_2gpu: PP=2 backward/training step
        - test_fsdp_pp_gradient_correctness_2gpu: PP=2 gradient correctness vs non-PP

    2 GPU Schedule Tests:
        - test_fsdp_pp_zbv_2gpu[forward_schedule]: PP=2, schedule.eval() with ZBVZeroBubble
        - test_fsdp_pp_zbv_2gpu[backward_schedule]: PP=2, schedule.step() with ZBVZeroBubble

    4 GPU Tests (Extended PP):
        - test_fsdp_pp_forward_4gpu: PP=4 forward pass
        - test_fsdp_pp_backward_4gpu: PP=4 backward pass

    4 GPU Combination Tests:
        - test_fsdp_pp_with_fsdp_sharding_4gpu: PP=2, DP=2 (FSDP + PP)
"""

import subprocess
import tempfile

import pytest
import torch

from areal.infra.platforms import current_platform
from areal.utils.network import find_free_ports

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

# Path to the torchrun test script (relative to project root)
_TORCHRUN_SCRIPT = "tests/fsdp/torchrun/run_fsdp_pp_tests.py"


def _run_pp_test_with_torchrun(
    script: str,
    n_gpus: int,
    extra_args: list[str] | None = None,
    timeout: int = 300,
):
    """Run a PP test script with torchrun.

    Args:
        script: Path to the test script.
        n_gpus: Number of GPUs to use.
        extra_args: Additional command line arguments.
        timeout: Timeout in seconds.

    Raises:
        pytest.fail: If the test fails, times out, or returns non-"Passed" result.
    """
    port = find_free_ports(1)[0]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        out_file = f.name

    cmd = [
        "torchrun",
        f"--nproc_per_node={n_gpus}",
        "--nnodes=1",
        "--master-addr=localhost",
        f"--master_port={port}",
        script,
        f"--output={out_file}",
    ]
    if extra_args:
        cmd.extend(extra_args)

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        # Check result file
        with open(out_file) as f:
            test_result = f.read().strip()

        if test_result != "Passed":
            pytest.fail(
                f"Test returned '{test_result}'. "
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )

    except subprocess.CalledProcessError as e:
        pytest.fail(f"Test failed with error: {e.stderr}\nstdout: {e.stdout}")
    except subprocess.TimeoutExpired:
        pytest.fail(f"Test timed out after {timeout} seconds")


# =============================================================================
# 2 GPU Tests (Core FSDP PP)
# =============================================================================


@pytest.mark.multi_gpu
def test_fsdp_pp_forward_2gpu():
    """Test FSDP PP forward pass with 2 GPUs (pp=2).

    Validates that a HuggingFace-style model split across 2 pipeline stages
    via pipeline_llm_hf produces the same output as the non-split golden model.
    Uses schedule.eval() with the 1F1B schedule.
    """
    if current_platform.device_count() < 2:
        pytest.skip("This test requires 2 GPUs")

    _run_pp_test_with_torchrun(
        _TORCHRUN_SCRIPT,
        n_gpus=2,
        extra_args=["--test_type=forward", "--pp_size=2"],
    )


@pytest.mark.multi_gpu
def test_fsdp_pp_backward_2gpu():
    """Test FSDP PP backward/training step with 2 GPUs (pp=2).

    Validates that gradients flow correctly through all pipeline stages
    after schedule.step(). Checks that every trainable parameter has a
    non-zero, non-NaN gradient.
    """
    if current_platform.device_count() < 2:
        pytest.skip("This test requires 2 GPUs")

    _run_pp_test_with_torchrun(
        _TORCHRUN_SCRIPT,
        n_gpus=2,
        extra_args=["--test_type=backward", "--pp_size=2"],
    )


@pytest.mark.multi_gpu
def test_fsdp_pp_gradient_correctness_2gpu():
    """Test FSDP PP gradient correctness with 2 GPUs (pp=2).

    Verifies that PP gradients match the gradients from a non-pipelined
    forward-backward pass on the same model with identical weights and inputs.
    Parameters that exist in both the PP model parts and the golden model
    are compared element-wise within a tolerance.
    """
    if current_platform.device_count() < 2:
        pytest.skip("This test requires 2 GPUs")

    _run_pp_test_with_torchrun(
        _TORCHRUN_SCRIPT,
        n_gpus=2,
        extra_args=["--test_type=gradient_correctness", "--pp_size=2"],
        timeout=120,
    )


# =============================================================================
# 2 GPU Schedule Tests (ZBVZeroBubble)
# =============================================================================


@pytest.mark.multi_gpu
@pytest.mark.parametrize("test_type", ["forward_schedule", "backward_schedule"])
def test_fsdp_pp_zbv_2gpu(test_type: str):
    """Test ZBVZeroBubble schedule with 2 GPUs (pp=2).

    Parametrized over forward_schedule (schedule.eval()) and
    backward_schedule (schedule.step()). ZBVZeroBubble uses V-style
    stage assignment where each rank holds 2 virtual stages.

    Args:
        test_type: Either "forward_schedule" or "backward_schedule".
    """
    if current_platform.device_count() < 2:
        pytest.skip("This test requires 2 GPUs")

    _run_pp_test_with_torchrun(
        _TORCHRUN_SCRIPT,
        n_gpus=2,
        extra_args=[
            f"--test_type={test_type}",
            "--pp_size=2",
            "--pp_schedule=ZBVZeroBubble",
        ],
    )


# =============================================================================
# 4 GPU Tests (Extended PP)
# =============================================================================


@pytest.mark.multi_gpu
def test_fsdp_pp_forward_4gpu():
    """Test FSDP PP forward pass with 4 GPUs (pp=4).

    Validates PP with 4 stages using a HuggingFace-style model.
    Uses schedule.eval() with the 1F1B schedule.
    """
    if current_platform.device_count() < 4:
        pytest.skip("This test requires 4 GPUs")

    _run_pp_test_with_torchrun(
        _TORCHRUN_SCRIPT,
        n_gpus=4,
        extra_args=["--test_type=forward", "--pp_size=4"],
    )


@pytest.mark.multi_gpu
def test_fsdp_pp_backward_4gpu():
    """Test FSDP PP backward pass with 4 GPUs (pp=4).

    Validates gradient flow through 4 pipeline stages using schedule.step().
    """
    if current_platform.device_count() < 4:
        pytest.skip("This test requires 4 GPUs")

    _run_pp_test_with_torchrun(
        _TORCHRUN_SCRIPT,
        n_gpus=4,
        extra_args=["--test_type=backward", "--pp_size=4"],
    )


# =============================================================================
# 4 GPU Combination Tests (FSDP + PP)
# =============================================================================


@pytest.mark.multi_gpu
def test_fsdp_pp_with_fsdp_sharding_4gpu():
    """Test FSDP sharding combined with PP on 4 GPUs (pp=2, dp=2).

    Uses a 2D device mesh with (pp=2, dp=2). Each PP stage's model part
    is wrapped with FSDP for data-parallel sharding across the dp dimension.
    Validates that gradients exist and are valid after schedule.step().
    """
    if current_platform.device_count() < 4:
        pytest.skip("This test requires 4 GPUs")

    _run_pp_test_with_torchrun(
        _TORCHRUN_SCRIPT,
        n_gpus=4,
        extra_args=["--test_type=fsdp_pp_sharding", "--pp_size=2"],
    )
