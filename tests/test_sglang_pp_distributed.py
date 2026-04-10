"""Distributed tests for sglang PP pipeline parallelism.

These tests verify that per-PP-rank NCCL weight update groups work correctly
in a multi-GPU distributed setting with DP=2, PP=2, TP=2.

Requires 4-8 GPUs to run.
"""
import subprocess
import pytest
from areal.api.alloc_mode import ModelAllocation
from areal.infra.platforms import current_platform
from areal.utils.network import find_free_ports


def _run_test_with_torchrun(
    alloc_mode: str, test_type: str, output: str, gen_pp_size: int = 1
):
    port = find_free_ports(1)[0]
    n_gpus = ModelAllocation.from_str(alloc_mode).parallel.world_size
    try:
        subprocess.run(
            [
                "torchrun",
                f"--nproc_per_node={n_gpus}",
                "--nnodes=1",
                "--master-addr=localhost",
                f"--master_port={port}",
                "tests/torchrun/run_sglang_pp_weight_sync.py",
                f"--backend={alloc_mode}",
                f"--output={output}",
                f"--test_type={test_type}",
                f"--gen_pp_size={gen_pp_size}",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Test failed:\nstdout: {e.stdout}\nstderr: {e.stderr}")
    except subprocess.TimeoutExpired as e:
        pytest.fail(f"Test timed out after 600s")
    with open(output) as f:
        result = f.read().strip()
    assert result == "Passed", f"Test failed: {result}"


@pytest.mark.multi_gpu
@pytest.mark.slow
@pytest.mark.sglang
def test_pp2_tp2_weight_group_init(tmp_path_factory):
    """Test per-PP-rank NCCL group initialization with PP=2, TP=2."""
    if current_platform.device_count() < 4:
        pytest.skip("Requires at least 4 GPUs")
    output = tmp_path_factory.mktemp("test_output") / "pp2_tp2_group_init.out"
    _run_test_with_torchrun(
        "megatron:d1p2t2", "group_init", str(output), gen_pp_size=2
    )


@pytest.mark.multi_gpu
@pytest.mark.slow
@pytest.mark.sglang
def test_dp2_pp2_tp2_weight_sync(tmp_path_factory):
    """End-to-end weight sync test with DP=2, PP=2, TP=2 (8 GPUs)."""
    if current_platform.device_count() < 8:
        pytest.skip("Requires 8 GPUs for DP=2, PP=2, TP=2")
    output = tmp_path_factory.mktemp("test_output") / "dp2_pp2_tp2_weight_sync.out"
    _run_test_with_torchrun(
        "megatron:d2p2t2", "weight_sync", str(output), gen_pp_size=2
    )


@pytest.mark.multi_gpu
@pytest.mark.slow
@pytest.mark.sglang
def test_pp1_backward_compatible(tmp_path_factory):
    """Verify PP=1 still works correctly (backward compatibility)."""
    if current_platform.device_count() < 4:
        pytest.skip("Requires at least 4 GPUs")
    output = tmp_path_factory.mktemp("test_output") / "pp1_backward.out"
    _run_test_with_torchrun(
        "megatron:d2p1t2", "weight_sync", str(output), gen_pp_size=1
    )
