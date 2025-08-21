import subprocess

import pytest

from areal.experimental.api.io_struct import AllocationMode
from areal.utils.network import find_free_ports


def _run_test_with_torchrun(model_type: str, alloc_mode: str, output: str):
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
                "areal/experimental/tests/torchrun/run_megatron_engine_distributed.py",
                f"--model_type={model_type}",
                f"--allocation_mode={alloc_mode}",
                f"--output={output}",
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Test failed with error: {e.stderr.decode()}")
    with open(output, "r") as f:
        result = f.read().strip()
    assert result == "Passed", f"Test failed: {result}"


@pytest.mark.two_gpu
def test_qwen3_tensor_parallel(tmp_path_factory):
    output = tmp_path_factory.mktemp("test_output") / "qwen3_tensor_parallel.out"
    _run_test_with_torchrun("qwen3", "d1p1t2", output=str(output))


@pytest.mark.two_gpu
def test_qwen3_pipeline_parallel(tmp_path_factory):
    output = tmp_path_factory.mktemp("test_output") / "qwen3_pipeline_parallel.out"
    _run_test_with_torchrun("qwen3", "d1p2t1", output=str(output))


@pytest.mark.two_gpu
def test_qwen3_context_parallel(tmp_path_factory):
    output = tmp_path_factory.mktemp("test_output") / "qwen3_context_parallel.out"
    _run_test_with_torchrun("qwen3", "d1p1t1c2", output=str(output))


@pytest.mark.two_gpu
def test_qwen3moe_expert_parallel(tmp_path_factory):
    output = tmp_path_factory.mktemp("test_output") / "qwen3moe_expert_parallel.out"
    _run_test_with_torchrun("qwen3moe", "d1p1t2c1/d1p1t1e2", output=str(output))
