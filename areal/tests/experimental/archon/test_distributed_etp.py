"""Expert Tensor Parallelism (ETP) tests for Archon Engine.

Tests ETP configurations where etp=tp (ExpertTensorParallel with 2D sharding).

Run tests:
    pytest areal/tests/experimental/archon/test_distributed_etp.py -v -m multi_gpu
"""

import subprocess

import pytest
import torch

from areal.platforms import current_platform
from areal.utils.network import find_free_ports

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


def _run_ep_test_with_torchrun(n_gpus: int, test_type: str, output: str):
    """Run EP/ETP test with torchrun.

    Args:
        n_gpus: Number of GPUs to use
        test_type: Type of test (etp_forward, etp_weight_sync, tp_only_forward)
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


# =============================================================================
# ETP Forward/Weight Sync Tests (ep>1, etp=tp) - Requires 4 GPUs
# =============================================================================


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_archon_etp_forward_4gpu(tmp_path_factory):
    """Test ETP forward numerical correctness (ep=2, tp=2, etp=2) on 4 GPUs.

    Tests ExpertTensorParallel with 2D weight sharding [Shard(0), Shard(1/2)].
    Verify: ETP model output matches non-EP golden model output.
    """
    if current_platform.device_count() < 4:
        pytest.skip("This test requires 4 GPUs")
    output = tmp_path_factory.mktemp("test_output") / "etp_forward.out"
    _run_ep_test_with_torchrun(4, "etp_forward", str(output))


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_archon_etp_weight_sync_4gpu(tmp_path_factory):
    """Test ETP weight gather and roundtrip (ep=2, tp=2, etp=2) on 4 GPUs.

    Tests weight gather/roundtrip with ExpertTensorParallel.
    Verify: Gathered weights match original, cross-rank consistency, roundtrip.
    """
    if current_platform.device_count() < 4:
        pytest.skip("This test requires 4 GPUs")
    output = tmp_path_factory.mktemp("test_output") / "etp_weight_sync.out"
    _run_ep_test_with_torchrun(4, "etp_weight_sync", str(output))


# =============================================================================
# TP-Only MoE Tests (ep=1, tp>1)
# =============================================================================


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_archon_tp_only_moe_forward_2gpu(tmp_path_factory):
    """Test TP-only forward for MoE experts (ep=1, tp=2) on 2 GPUs.

    Tests TensorParallel class for MoE experts when EP is disabled.
    Verify: TP-only MoE model output matches non-TP golden model output.
    """
    if current_platform.device_count() < 2:
        pytest.skip("This test requires 2 GPUs")
    output = tmp_path_factory.mktemp("test_output") / "tp_only_forward.out"
    _run_ep_test_with_torchrun(2, "tp_only_forward", str(output))
