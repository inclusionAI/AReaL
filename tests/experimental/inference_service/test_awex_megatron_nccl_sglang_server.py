"""Controller tests for awex Megatron <-> AReaL SGLang NCCL integration.

Pattern follows Archon distributed tests:
1) A generic torchrun script contains SPMD logic.
2) This file acts as controller that launches torchrun with different parallel
   allocations and verifies the output marker file.
"""

from __future__ import annotations

import os
import subprocess
import sys
import uuid
from contextlib import contextmanager
from dataclasses import asdict

import pytest
import torch

from tests.experimental.inference_service.integration_utils import get_test_model_path

from areal.api.cli_args import OpenAIProxyConfig, SchedulingSpec, SGLangConfig
from areal.experimental.inference_service.controller.config import (
    GatewayControllerConfig,
)
from areal.experimental.inference_service.controller.controller import (
    GatewayInferenceController,
)
from areal.infra.platforms import current_platform
from areal.infra.scheduler.local import LocalScheduler
from areal.utils.network import find_free_ports

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


def _visible_devices() -> list[int]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if raw:
        return [int(x.strip()) for x in raw.split(",") if x.strip()]
    return list(range(current_platform.device_count()))


def _split_disjoint_gpus(
    total_required: int, trainer_count: int
) -> tuple[list[int], list[int]]:
    visible = _visible_devices()
    if len(visible) < total_required:
        pytest.skip(
            f"This test requires at least {total_required} visible GPUs, got {len(visible)}"
        )
    if trainer_count <= 0 or trainer_count >= total_required:
        raise ValueError(
            f"Invalid trainer_count={trainer_count}; must be in [1, {total_required - 1}]"
        )

    pool = visible[:total_required]
    trainer = pool[:trainer_count]
    inference = pool[trainer_count:]
    if set(trainer).intersection(inference):
        raise AssertionError(
            f"Trainer/inference GPUs overlap: trainer={trainer}, inference={inference}"
        )
    return trainer, inference


def _start_controller(
    *,
    inf_tp: int,
    tmp_root: str,
    inference_gpus: list[int],
    model_path: str,
) -> tuple[GatewayInferenceController, LocalScheduler]:
    trial_name = f"awex-inf-{uuid.uuid4().hex[:8]}"
    nfs_root = os.path.join(tmp_root, "name_resolve")
    os.makedirs(nfs_root, exist_ok=True)

    scheduler = LocalScheduler(
        gpu_devices=inference_gpus,
        experiment_name="awex_inference_service_test",
        trial_name=trial_name,
        fileroot=tmp_root,
        nfs_record_root=nfs_root,
        name_resolve_type="nfs",
    )

    inf_dp = len(inference_gpus) // inf_tp
    backend = f"sglang:d{inf_dp}t{inf_tp}"

    controller_cfg = GatewayControllerConfig(
        tokenizer_path=model_path,
        model_path=model_path,
        backend=backend,
        scheduling_spec=(SchedulingSpec(cpu=2, mem=8, gpu=1, port_count=2),),
        setup_timeout=300.0,
        request_timeout=300.0,
        openai=OpenAIProxyConfig(admin_api_key="awex-test-admin"),
    )

    controller = GatewayInferenceController(config=controller_cfg, scheduler=scheduler)
    controller.initialize(
        role="rollout",
        server_args=asdict(
            SGLangConfig(
                skip_tokenizer_init=True,
                model_path=model_path,
                mem_fraction_static=0.15,
                launch_server_module="areal.engine.sglang_ext.areal_sglang_server",
            )
        ),
    )
    if not controller.inference_worker_addrs:
        controller.destroy()
        scheduler.delete_workers(None)
        raise RuntimeError(
            "Controller started but no inference worker addrs are available"
        )

    return controller, scheduler


def _run_awex_sglang_torchrun(
    trainer_gpu_ids: list[int],
    dp_size: int,
    tp_size: int,
    pp_size: int,
    infer_tp_size: int,
    awex_server_urls: list[str],
    output: str,
    *,
    expect_success: bool,
    expected_error_substr: str | None = None,
):
    model_path = get_test_model_path()
    torchrun_port = find_free_ports(1)[0]
    cmd = [
        "torchrun",
        f"--nproc_per_node={len(trainer_gpu_ids)}",
        "--nnodes=1",
        "--master-addr=localhost",
        f"--master-port={torchrun_port}",
        "tests/experimental/inference_service/torchrun/run_awex_megatron_sglang_nccl.py",
        f"--dp-size={dp_size}",
        f"--tp-size={tp_size}",
        f"--pp-size={pp_size}",
        f"--infer-tp-size={infer_tp_size}",
        f"--model-path={model_path}",
        f"--awex-server-urls={','.join(awex_server_urls)}",
        f"--output={output}",
        "--health-timeout=240",
        "--rpc-timeout=240",
    ]
    environ = os.environ.copy()
    # AWEX reader/writer communicators span independently launched process trees.
    # In some CI/container setups, those processes do not share a stable /dev/shm
    # namespace; NCCL SHM transport can fail to attach segments and deadlock first
    # collectives. Force NCCL to avoid SHM for this integration test.
    # environ["NCCL_SHM_DISABLE"] = "1"
    environ["NCCL_CUMEM_ENABLE"] = "0"
    environ["NCCL_NVLS_ENABLE"] = "0"
    # DETAIL may wrap process groups and has known device-binding caveats in
    # some PyTorch versions; INFO keeps useful logs without that extra wrapper.
    environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
    environ["TORCH_NCCL_ENABLE_MONITORING"] = "0"
    environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, trainer_gpu_ids))
    completed: subprocess.CompletedProcess[str] | None = None
    try:
        completed = subprocess.run(
            cmd,
            env=environ,
            check=False,
            stdout=sys.stdout,
            stderr=sys.stderr,
            text=True,
            timeout=1200,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(f"Torchrun timed out after {exc.timeout}s: {' '.join(cmd)}")
    if completed is None:
        pytest.fail(f"Torchrun failed to execute command: {' '.join(cmd)}")
    assert completed is not None

    result = ""
    if os.path.exists(output):
        with open(output, encoding="utf-8") as f:
            result = f.read().strip()

    if expect_success:
        if completed.returncode != 0:
            pytest.fail(
                f"Torchrun failed with exit code {completed.returncode}. "
                f"See streamed logs above. Command: {' '.join(cmd)}"
            )
        assert result == "Passed", f"Integration failed: {result}"
        return

    if completed.returncode == 0 and result == "Passed":
        pytest.fail(
            "Expected this topology to be rejected by AWEX, but torchrun passed. "
            f"Command: {' '.join(cmd)}"
        )
    if expected_error_substr:
        assert expected_error_substr in result, (
            "Torchrun failed as expected, but missing expected error marker. "
            f"expected={expected_error_substr!r} actual={result!r}"
        )


@contextmanager
def _temporary_env(overrides: dict[str, str]):
    previous: dict[str, str | None] = {}
    for key, value in overrides.items():
        previous[key] = os.environ.get(key)
        os.environ[key] = value
    try:
        yield
    finally:
        for key, old in previous.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


@pytest.mark.multi_gpu
@pytest.mark.slow
@pytest.mark.parametrize(
    "split_name,trainer_gpu_count,dp_size,tp_size,pp_size,inf_tp,expect_success,expected_error_substr",
    [
        ("4plus4_d4_t4", 4, 4, 1, 1, 4, True, None),
        ("4plus4_d2t2_t4", 4, 2, 2, 1, 4, True, None),
        ("4plus4_d4_d4", 4, 4, 1, 1, 1, True, None),
        ("4plus4_p4_t4", 4, 1, 1, 4, 4, True, None),
    ],
)
def test_awex_megatron_sglang_nccl_disjoint_gpu_split(
    tmp_path_factory,
    split_name: str,
    trainer_gpu_count: int,
    dp_size: int,
    tp_size: int,
    pp_size: int,
    inf_tp: int,
    expect_success: bool,
    expected_error_substr: str | None,
):
    """Run awex Megatron<->SGLang with trainer/inference disjoint GPU pools.

    The test controller launches inference workers via GatewayInferenceController.
    The trainer torchrun process only handles writer/meta-server logic and talks to
    the direct inference worker URL exposed by the controller.
    """

    total_required_gpus = 8
    expected_trainer_world_size = dp_size * tp_size * pp_size
    if trainer_gpu_count != expected_trainer_world_size:
        raise ValueError(
            f"Invalid trainer setup for {split_name}: trainer_gpu_count={trainer_gpu_count} "
            f"!= dp*tp*pp={expected_trainer_world_size}"
        )

    trainer_gpu_ids, inference_gpu_ids = _split_disjoint_gpus(
        total_required=total_required_gpus,
        trainer_count=trainer_gpu_count,
    )

    model_path = get_test_model_path()

    output = (
        tmp_path_factory.mktemp("test_output") / f"awex_sglang_nccl_{split_name}.out"
    )
    runtime_root = str(tmp_path_factory.mktemp(f"runtime_{split_name}"))

    controller = None
    scheduler = None
    with _temporary_env(
        {
            # "NCCL_SHM_DISABLE": "1",
            "NCCL_CUMEM_ENABLE": "0",
            "NCCL_NVLS_ENABLE": "0",
            "TORCH_DISTRIBUTED_DEBUG": "INFO",
            "TORCH_NCCL_ENABLE_MONITORING": "0",
        }
    ):
        try:
            controller, scheduler = _start_controller(
                inf_tp=inf_tp,
                tmp_root=runtime_root,
                inference_gpus=inference_gpu_ids,
                model_path=model_path,
            )
            awex_server_urls = controller.inference_worker_addrs
            if not awex_server_urls:
                raise RuntimeError("Controller started with no inference worker URLs")

            _run_awex_sglang_torchrun(
                trainer_gpu_ids=trainer_gpu_ids,
                dp_size=dp_size,
                tp_size=tp_size,
                pp_size=pp_size,
                infer_tp_size=inf_tp,
                awex_server_urls=awex_server_urls,
                output=str(output),
                expect_success=expect_success,
                expected_error_substr=expected_error_substr,
            )
        finally:
            if controller is not None:
                controller.destroy()
            if scheduler is not None:
                scheduler.delete_workers(None)
