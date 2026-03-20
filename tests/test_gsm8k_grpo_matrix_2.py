import os
import shutil
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
    scheduler_type: str,
    timeout: int,
    fileroot: str,
    nfs_record_root: str,
    n_gpus_per_node: int,
):
    model_path = get_model_path("/storage/openpsi/models/Qwen__Qwen3-0.6B", "Qwen/Qwen3-0.6B")
    dataset_path = get_dataset_path("/storage/openpsi/data/gsm8k", "openai/gsm8k")

    example_file = "examples/math/gsm8k_rl.py"
    config_name = "examples/math/gsm8k_grpo.yaml"

    run_id = uuid.uuid4().hex[:8]
    now = int(time.time())
    additional_args = [
        f"scheduler.type={scheduler_type}",
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
        f"cluster.n_gpus_per_node={n_gpus_per_node}",
        f"cluster.fileroot={fileroot}",
        f"cluster.name_resolve.nfs_record_root={nfs_record_root}",
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


def _slurm_env_ready() -> tuple[bool, str]:
    if os.getenv("AREAL_RUN_SLURM_SMOKE") != "1":
        return False, "Set AREAL_RUN_SLURM_SMOKE=1 to enable Slurm smoke tests"
    if shutil.which("sbatch") is None:
        return False, "sbatch not found in PATH"
    fileroot = os.getenv("AREAL_SLURM_SHARED_FILEROOT")
    nfs_root = os.getenv("AREAL_SLURM_SHARED_NFS_RECORD_ROOT")
    if not fileroot or not nfs_root:
        return (
            False,
            "Set AREAL_SLURM_SHARED_FILEROOT and AREAL_SLURM_SHARED_NFS_RECORD_ROOT",
        )
    if not os.path.isdir(fileroot) or not os.path.isdir(nfs_root):
        return False, "Slurm shared paths must exist and be directories"
    return True, ""


def _run_matrix_smoke(
    tmp_path_factory,
    *,
    train_backend: str,
    inf_backend: str,
    scheduler_type: str,
    timeout: int,
):
    if inf_backend not in {"sglang", "vllm"}:
        raise ValueError(f"Unsupported inf_backend: {inf_backend}")
    alloc_mode = f"{inf_backend}:d1p1t1+{train_backend}:d1p1t1"

    if scheduler_type == "slurm":
        ok, reason = _slurm_env_ready()
        if not ok:
            pytest.skip(reason)
        fileroot = os.environ["AREAL_SLURM_SHARED_FILEROOT"]
        nfs_root = os.environ["AREAL_SLURM_SHARED_NFS_RECORD_ROOT"]
        n_gpus_per_node = int(os.getenv("AREAL_SLURM_N_GPUS_PER_NODE", "1"))
        return _run_gsm8k_grpo_smoke(
            tmp_path_factory,
            allocation_mode=alloc_mode,
            scheduler_type="slurm",
            timeout=timeout,
            fileroot=fileroot,
            nfs_record_root=nfs_root,
            n_gpus_per_node=n_gpus_per_node,
        )

    if scheduler_type == "ray":
        pytest.importorskip("ray")

    fileroot = str(tmp_path_factory.mktemp("experiments"))
    nfs_root = str(tmp_path_factory.mktemp("name_resolve"))
    return _run_gsm8k_grpo_smoke(
        tmp_path_factory,
        allocation_mode=alloc_mode,
        scheduler_type=scheduler_type,
        timeout=timeout,
        fileroot=fileroot,
        nfs_record_root=nfs_root,
        n_gpus_per_node=2,
    )


@pytest.mark.slow
@pytest.mark.sglang
@pytest.mark.multi_gpu
@pytest.mark.parametrize(
    "train_backend",
    ["fsdp", "megatron", "archon"],
)
def test_gsm8k_grpo_smoke_sglang_matrix(tmp_path_factory, train_backend):
    success = _run_matrix_smoke(
        tmp_path_factory,
        train_backend=train_backend,
        inf_backend="sglang",
        scheduler_type="local",
        timeout=1800,
    )
    assert success, f"GSM8K GRPO smoke failed: sglang+d1 + {train_backend}+d1 (local)"


@pytest.mark.slow
@pytest.mark.vllm
@pytest.mark.multi_gpu
@pytest.mark.parametrize(
    "train_backend",
    ["fsdp", "megatron", "archon"],
)
def test_gsm8k_grpo_smoke_vllm_matrix(tmp_path_factory, train_backend):
    success = _run_matrix_smoke(
        tmp_path_factory,
        train_backend=train_backend,
        inf_backend="vllm",
        scheduler_type="local",
        timeout=1800,
    )
    assert success, f"GSM8K GRPO smoke failed: vllm+d1 + {train_backend}+d1 (local)"


@pytest.mark.slow
@pytest.mark.sglang
@pytest.mark.multi_gpu
@pytest.mark.parametrize("train_backend", ["fsdp", "megatron", "archon"])
def test_gsm8k_grpo_smoke_sglang_matrix_ray(tmp_path_factory, train_backend):
    success = _run_matrix_smoke(
        tmp_path_factory,
        train_backend=train_backend,
        inf_backend="sglang",
        scheduler_type="ray",
        timeout=1800,
    )
    assert success, f"GSM8K GRPO smoke failed: sglang+d1 + {train_backend}+d1 (ray)"


@pytest.mark.slow
@pytest.mark.vllm
@pytest.mark.multi_gpu
@pytest.mark.parametrize("train_backend", ["fsdp", "megatron", "archon"])
def test_gsm8k_grpo_smoke_vllm_matrix_ray(tmp_path_factory, train_backend):
    success = _run_matrix_smoke(
        tmp_path_factory,
        train_backend=train_backend,
        inf_backend="vllm",
        scheduler_type="ray",
        timeout=1800,
    )
    assert success, f"GSM8K GRPO smoke failed: vllm+d1 + {train_backend}+d1 (ray)"


@pytest.mark.slow
@pytest.mark.sglang
@pytest.mark.multi_gpu
@pytest.mark.parametrize("train_backend", ["fsdp", "megatron", "archon"])
def test_gsm8k_grpo_smoke_sglang_matrix_slurm(tmp_path_factory, train_backend):
    success = _run_matrix_smoke(
        tmp_path_factory,
        train_backend=train_backend,
        inf_backend="sglang",
        scheduler_type="slurm",
        timeout=3600,
    )
    assert success, f"GSM8K GRPO smoke failed: sglang+d1 + {train_backend}+d1 (slurm)"


@pytest.mark.slow
@pytest.mark.vllm
@pytest.mark.multi_gpu
@pytest.mark.parametrize("train_backend", ["fsdp", "megatron", "archon"])
def test_gsm8k_grpo_smoke_vllm_matrix_slurm(tmp_path_factory, train_backend):
    success = _run_matrix_smoke(
        tmp_path_factory,
        train_backend=train_backend,
        inf_backend="vllm",
        scheduler_type="slurm",
        timeout=3600,
    )
    assert success, f"GSM8K GRPO smoke failed: vllm+d1 + {train_backend}+d1 (slurm)"
