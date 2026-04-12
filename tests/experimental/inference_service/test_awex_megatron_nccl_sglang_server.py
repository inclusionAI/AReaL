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

    infer_tp = len(inference_gpus)
    backend = f"sglang:d1t{infer_tp}"

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
    awex_server_url: str,
    output: str,
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
        f"--model-path={model_path}",
        f"--awex-server-url={awex_server_url}",
        f"--output={output}",
        "--health-timeout=240",
        "--rpc-timeout=240",
    ]
    environ = os.environ.copy()
    environ["NCCL_P2P_DISABLE"] = "1"
    environ["NCCL_CUMEM_ENABLE"] = "0"
    environ["NCCL_NVLS_ENABLE"] = "0"
    environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    environ["TORCH_NCCL_ENABLE_MONITORING"] = "0"
    environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, trainer_gpu_ids))
    try:
        subprocess.run(
            cmd,
            env=environ,
            check=True,
            stdout=sys.stdout,
            stderr=sys.stderr,
            text=True,
            timeout=1200,
        )
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
        if isinstance(exc, subprocess.TimeoutExpired):
            pytest.fail(f"Torchrun timed out after {exc.timeout}s: {' '.join(cmd)}")
        else:
            pytest.fail(
                f"Torchrun failed with exit code {exc.returncode}. "
                f"See streamed logs above. Command: {' '.join(cmd)}"
            )

    with open(output, encoding="utf-8") as f:
        result = f.read().strip()
    assert result == "Passed", f"Integration failed: {result}"


@pytest.mark.multi_gpu
@pytest.mark.slow
@pytest.mark.parametrize(
    "split_name,trainer_gpu_count,dp_size,tp_size,pp_size",
    [
        ("2plus6", 2, 2, 1, 1),
        ("4plus4", 4, 1, 2, 2),
    ],
)
def test_awex_megatron_sglang_nccl_disjoint_gpu_split(
    tmp_path_factory,
    split_name: str,
    trainer_gpu_count: int,
    dp_size: int,
    tp_size: int,
    pp_size: int,
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
    try:
        controller, scheduler = _start_controller(
            tmp_root=runtime_root,
            inference_gpus=inference_gpu_ids,
            model_path=model_path,
        )
        awex_server_url = controller.inference_worker_addrs[0]

        _run_awex_sglang_torchrun(
            trainer_gpu_ids=trainer_gpu_ids,
            dp_size=dp_size,
            tp_size=tp_size,
            pp_size=pp_size,
            awex_server_url=awex_server_url,
            output=str(output),
        )
    finally:
        if controller is not None:
            controller.destroy()
        if scheduler is not None:
            scheduler.delete_workers(None)
