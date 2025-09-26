import subprocess

import pytest

from areal.api.alloc_mode import AllocationMode
from areal.platforms import current_platform
from areal.utils.network import find_free_ports


def _run_test_with_torchrun(
    model_type: str, alloc_mode: str, test_type: str, output: str
):
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
                f"--test_type={test_type}",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Test failed with error: {e.stderr}")
    with open(output, "r") as f:
        result = f.read().strip()
    assert result == "Passed", f"Test failed: {result}"


@pytest.mark.multi_gpu
def test_qwen3_tensor_parallel(tmp_path_factory):
    if current_platform.device_count() < 2:
        pytest.skip("tensor parallel requires 2 GPUs to run")
    output = tmp_path_factory.mktemp("test_output") / "qwen3_tensor_parallel.out"
    _run_test_with_torchrun("qwen3", "d1p1t2", test_type="forward", output=str(output))


@pytest.mark.multi_gpu
def test_qwen3_pipeline_parallel(tmp_path_factory):
    if current_platform.device_count() < 2:
        pytest.skip("pipeline parallel requires 2 GPUs to run")
    output = tmp_path_factory.mktemp("test_output") / "qwen3_pipeline_parallel.out"
    _run_test_with_torchrun("qwen3", "d1p2t1", test_type="forward", output=str(output))


@pytest.mark.multi_gpu
def test_qwen3_context_parallel(tmp_path_factory):
    if current_platform.device_count() < 2:
        pytest.skip("context parallel requires 2 GPUs to run")
    output = tmp_path_factory.mktemp("test_output") / "qwen3_context_parallel.out"
    _run_test_with_torchrun(
        "qwen3", "d1p1t1c2", test_type="forward", output=str(output)
    )


@pytest.mark.multi_gpu
def test_qwen3moe_expert_parallel(tmp_path_factory):
    if current_platform.device_count() < 4:
        pytest.skip("Qwen3 MoE expert parallel requires 4 GPUs to run")
    output = tmp_path_factory.mktemp("test_output") / "qwen3moe_expert_parallel.out"
    _run_test_with_torchrun(
        "qwen3moe",
        "attn:d1p1t2c2|ffn:d1p1t1e4",
        test_type="forward",
        output=str(output),
    )


@pytest.mark.multi_gpu
def test_qwen3_dcp_save_load(tmp_path_factory):
    if current_platform.device_count() < 8:
        pytest.skip("DCP save load requires 8 GPUs to run")
    output = tmp_path_factory.mktemp("test_output") / "qwen3_save_load.out"
    _run_test_with_torchrun(
        "qwen3",
        "d2p2t2",
        test_type="train_dcp_save_load",
        output=str(output),
    )


@pytest.mark.multi_gpu
def test_qwen3moe_dcp_save_load(tmp_path_factory):
    if current_platform.device_count() < 8:
        pytest.skip("Qwen3 MoE DCP save load requires 8 GPUs to run")
    output = tmp_path_factory.mktemp("test_output") / "qwen3moe_save_load.out"
    _run_test_with_torchrun(
        "qwen3moe",
        "attn:d1p1t4c2|ffn:d1p1t2e4",
        test_type="simple_dcp_save_load",
        output=str(output),
    )
