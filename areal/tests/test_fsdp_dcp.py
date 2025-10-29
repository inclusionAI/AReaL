import subprocess

import pytest

from areal.api.alloc_mode import AllocationMode
from areal.platforms import current_platform
from areal.utils.network import find_free_ports


def _run_test_with_torchrun(test_type: str, alloc_mode: str, output: str):
    port = find_free_ports(1)[0]
    n_gpus = AllocationMode.from_str(alloc_mode).train.world_size
    try:
        subprocess.run(
            [
                "torchrun",
                f"--nproc_per_node={n_gpus}",
                "--nnodes=1",
                "--master-addr=localhost",
                f"--master_port={port}",
                "areal/tests/torchrun/run_fsdp_dcp_distributed.py",
                f"--allocation_mode={alloc_mode}",
                f"--output={output}",
                f"--test_type={test_type}",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Test failed with error: {e.stderr}")
    with open(output) as f:
        result = f.read().strip()
    assert result == "Passed", f"Test failed: {result}"


@pytest.mark.multi_gpu
def test_fsdp_dcp_distributed_simple(tmp_path_factory):
    if current_platform.device_count() < 2:
        pytest.skip("Distributed test requires 2 GPUs to run")
    output = tmp_path_factory.mktemp("test_output") / "fsdp_dcp_simple_distributed.out"
    _run_test_with_torchrun("simple_dcp_save_load", "d2t1c1", str(output))


@pytest.mark.multi_gpu
def test_fsdp_dcp_distributed_train(tmp_path_factory):
    if current_platform.device_count() < 2:
        pytest.skip("Distributed test requires 2 GPUs to run")
    output = tmp_path_factory.mktemp("test_output") / "fsdp_dcp_train_distributed.out"
    _run_test_with_torchrun("train_dcp_save_load", "d2t1c1", str(output))


@pytest.mark.multi_gpu
def test_fsdp_dcp_distributed_forward(tmp_path_factory):
    if current_platform.device_count() < 2:
        pytest.skip("Distributed test requires 2 GPUs to run")
    output = tmp_path_factory.mktemp("test_output") / "fsdp_dcp_forward_distributed.out"
    _run_test_with_torchrun("forward", "d2t1c1", str(output))
