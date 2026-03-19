import time
import uuid

import pytest

from tests.test_examples import run_example
from tests.utils import get_dataset_path, get_model_path

from areal.infra.utils.concurrent import run_async_task


def _run_gsm8k_grpo_smoke(
    tmp_path_factory,
    *,
    allocation_mode: str,
    timeout: int,
):
    experiments_path = tmp_path_factory.mktemp("experiments")
    name_resolve_path = tmp_path_factory.mktemp("name_resolve")
    model_path = get_model_path("/storage/openpsi/models/Qwen__Qwen3-0.6B", "Qwen/Qwen3-0.6B")
    dataset_path = get_dataset_path("/storage/openpsi/data/gsm8k", "openai/gsm8k")

    example_file = "examples/math/gsm8k_rl.py"
    config_name = "examples/math/gsm8k_grpo.yaml"

    run_id = uuid.uuid4().hex[:8]
    now = int(time.time())
    additional_args = [
        "scheduler.type=local",
        f"experiment_name=smoke_ipv6_{run_id}",
        f"trial_name=t{now}_{run_id}",
        f"allocation_mode={allocation_mode}",
        "gconfig.n_samples=1",
        "gconfig.max_new_tokens=64",
        "actor.mb_spec.max_tokens_per_mb=1024",
        "train_dataset.batch_size=8",
        "valid_dataset.batch_size=8",
        f"train_dataset.path={dataset_path}",
        f"valid_dataset.path={dataset_path}",
        "cluster.n_gpus_per_node=2",
        f"cluster.fileroot={str(experiments_path)}",
        f"cluster.name_resolve.nfs_record_root={str(name_resolve_path)}",
        f"actor.path={model_path}",
    ]
    return run_async_task(
        run_example,
        example_file,
        config_name,
        *additional_args,
        timeout=timeout,
        single_controller=True,
    )


@pytest.mark.slow
@pytest.mark.sglang
@pytest.mark.multi_gpu
@pytest.mark.parametrize(
    "train_backend",
    ["fsdp", "megatron", "archon"],
)
def test_gsm8k_grpo_smoke_sglang_matrix(tmp_path_factory, train_backend):
    alloc_mode = f"sglang:d1p1t1+{train_backend}:d1p1t1"
    success = _run_gsm8k_grpo_smoke(
        tmp_path_factory,
        allocation_mode=alloc_mode,
        timeout=1800,
    )
    assert success, f"GSM8K GRPO smoke failed: {alloc_mode}"


@pytest.mark.slow
@pytest.mark.vllm
@pytest.mark.multi_gpu
@pytest.mark.parametrize(
    "train_backend",
    ["fsdp", "megatron", "archon"],
)
def test_gsm8k_grpo_smoke_vllm_matrix(tmp_path_factory, train_backend):
    alloc_mode = f"vllm:d1p1t1+{train_backend}:d1p1t1"
    success = _run_gsm8k_grpo_smoke(
        tmp_path_factory,
        allocation_mode=alloc_mode,
        timeout=1800,
    )
    assert success, f"GSM8K GRPO smoke failed: {alloc_mode}"
