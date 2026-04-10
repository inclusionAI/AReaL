#!/usr/bin/env python3
"""Torchrun entrypoint for awex Megatron <-> AReaL SGLang NCCL integration.

This script is intentionally generic over training-side parallel allocations and can
be launched by a controller pytest file with different strategies.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
import traceback

import requests
import torch
import torch.distributed as dist

from tests.experimental.inference_service.integration_utils import check_server_health


def _write_result(path: str, ok: bool, message: str = "") -> None:
    if dist.get_rank() != 0:
        return
    payload = "Passed" if ok else f"Failed: {message}"
    with open(path, "w", encoding="utf-8") as f:
        f.write(payload)


def _visible_devices() -> list[str]:
    env_name = "CUDA_VISIBLE_DEVICES"
    if env_name in os.environ and os.environ[env_name].strip():
        return [d.strip() for d in os.environ[env_name].split(",") if d.strip()]
    if torch.cuda.is_available():
        return [str(i) for i in range(torch.cuda.device_count())]
    return []


def _server_command(model_path: str, host: str, port: int, dist_port: int) -> list[str]:
    from areal.api.cli_args import SGLangConfig

    args = SGLangConfig.build_args(
        sglang_config=SGLangConfig(
            skip_tokenizer_init=True,
            model_path=model_path,
            mem_fraction_static=0.15,
        ),
        host=host,
        port=port,
        tp_size=1,
        base_gpu_id=0,
        dist_init_addr=f"{host}:{dist_port}",
    )
    cmd = [sys.executable, "-m", "areal.engine.sglang_ext.areal_sglang_server"]
    for key, value in args.items():
        if value is None:
            continue
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(value)])
    return cmd


def _init_megatron_parallel(tp_size: int, pp_size: int) -> None:
    from megatron.core import parallel_state as mpu
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    if not mpu.model_parallel_is_initialized():
        mpu.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=pp_size,
            virtual_pipeline_model_parallel_size=None,
            context_parallel_size=1,
            expert_model_parallel_size=1,
            create_gloo_process_groups=False,
        )
        model_parallel_cuda_manual_seed(42)


def _validate_allocation(
    world_size: int, dp_size: int, tp_size: int, pp_size: int
) -> None:
    expected = dp_size * tp_size * pp_size
    if expected != world_size:
        raise ValueError(
            f"Invalid allocation: dp({dp_size})*tp({tp_size})*pp({pp_size})={expected} "
            f"!= world_size({world_size})"
        )


def main(args: argparse.Namespace) -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    _validate_allocation(world_size, args.dp_size, args.tp_size, args.pp_size)

    visible = _visible_devices()
    if len(visible) < world_size:
        raise RuntimeError(
            f"Need >= {world_size} visible GPUs for torchrun world size {world_size}, got {len(visible)}"
        )

    from awex.engine.mcore import MegatronEngine
    from awex.meta.meta_server import start_meta_server, stop_meta_server
    from awex.tests.test_utils import megatron_model_from_hf
    from awex.util import device as device_util

    from areal.infra.utils.proc import kill_process_tree
    from areal.utils import network

    # Training ranks occupy GPUs [0..world_size-1], SGLang server occupies GPU [world_size].
    os.environ["DEVICE"] = str(local_rank)
    device_util.set_device(local_rank)

    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    server_proc: subprocess.Popen | None = None
    meta_started = False
    try:
        if rank == 0:
            meta_ip, meta_port = start_meta_server()
            meta_started = True
            os.environ["AWEX_META_ADDR_BCAST"] = f"{meta_ip}:{meta_port}"
            host = network.gethostip()
            sglang_port, sglang_dist_port = network.find_free_ports(2)
            os.environ["AWEX_SGLANG_HOST_BCAST"] = host
            os.environ["AWEX_SGLANG_PORT_BCAST"] = str(sglang_port)

            cmd = _server_command(args.model_path, host, sglang_port, sglang_dist_port)
            server_env = os.environ.copy()
            # Prefer a non-overlapping GPU when available; otherwise colocate.
            server_gpu_idx = world_size if world_size < len(visible) else 0
            server_env["CUDA_VISIBLE_DEVICES"] = visible[server_gpu_idx]
            server_proc = subprocess.Popen(
                cmd,
                env=server_env,
                stdout=sys.stdout,
                stderr=sys.stdout,
            )

        # Broadcast meta/server endpoint to all ranks via env + object list.
        shared = [
            os.environ.get("AWEX_META_ADDR_BCAST", ""),
            os.environ.get("AWEX_SGLANG_HOST_BCAST", ""),
            os.environ.get("AWEX_SGLANG_PORT_BCAST", ""),
        ]
        dist.broadcast_object_list(shared, src=0)
        meta_addr, sglang_host, sglang_port_str = shared
        sglang_port = int(sglang_port_str)
        base_url = f"http://{sglang_host}:{sglang_port}"

        init_result: dict[str, object] = {"ok": False, "error": None}
        update_result: dict[str, object] = {"ok": False, "error": None}

        def _call_awex_init() -> None:
            try:
                resp = requests.post(
                    f"{base_url}/areal_awex_init",
                    json={
                        "meta_server_addr": meta_addr,
                        "engine_rank": 0,
                        "num_engines": 1,
                        "comm_backend": "nccl",
                        "nnodes": 1,
                        "node_rank": 0,
                    },
                    timeout=900,
                )
                if resp.status_code == 200 and resp.json().get("success"):
                    init_result["ok"] = True
                else:
                    init_result["error"] = (
                        f"init failed: {resp.status_code} {resp.text}"
                    )
            except Exception as exc:  # pragma: no cover
                init_result["error"] = str(exc)

        def _call_awex_update(step_id: int) -> None:
            try:
                resp = requests.post(
                    f"{base_url}/areal_awex_update",
                    json={"step_id": step_id, "kwargs": {}},
                    timeout=900,
                )
                if resp.status_code == 200 and resp.json().get("success"):
                    update_result["ok"] = True
                else:
                    update_result["error"] = (
                        f"update failed: {resp.status_code} {resp.text}"
                    )
            except Exception as exc:  # pragma: no cover
                update_result["error"] = str(exc)

        if rank == 0:
            deadline = time.time() + 300
            while time.time() < deadline:
                if check_server_health(base_url):
                    break
                time.sleep(1)
            else:
                raise RuntimeError("AReaL SGLang server did not become healthy")

            init_thread = threading.Thread(target=_call_awex_init, daemon=True)
            init_thread.start()
        else:
            init_thread = None

        _init_megatron_parallel(tp_size=args.tp_size, pp_size=args.pp_size)
        model, hf_config = megatron_model_from_hf(
            model_path=args.model_path, use_mbridge=True
        )
        megatron_model = model[0] if isinstance(model, list) else model
        train_config = {
            "meta_server_addr": meta_addr,
            "comm_backend": "nccl",
            "enable_debug_mode": False,
            "tensor_model_parallel_size": args.tp_size,
            "pipeline_model_parallel_size": args.pp_size,
            "expert_model_parallel_size": 1,
        }
        megatron_engine = MegatronEngine(train_config, hf_config, megatron_model)
        megatron_engine.initialize()

        if rank == 0 and init_thread is not None:
            init_thread.join(timeout=900)
            if not init_result["ok"]:
                raise RuntimeError(f"Reader init failed: {init_result['error']}")

        dist.barrier()

        step_id = 1
        if rank == 0:
            update_thread = threading.Thread(
                target=_call_awex_update, kwargs={"step_id": step_id}, daemon=True
            )
            update_thread.start()
            time.sleep(1)
        else:
            update_thread = None

        megatron_engine.set_global_step(step_id)
        megatron_engine.write_weights()

        if rank == 0 and update_thread is not None:
            update_thread.join(timeout=900)
            if not update_result["ok"]:
                raise RuntimeError(f"Reader update failed: {update_result['error']}")

        dist.barrier()
        _write_result(args.output, True)
    except Exception as exc:
        tb = traceback.format_exc(limit=5)
        _write_result(args.output, False, f"{exc}\n{tb}")
        raise
    finally:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        if rank == 0:
            if server_proc is not None:
                kill_process_tree(server_proc.pid, graceful=True)
            if meta_started:
                stop_meta_server()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dp-size", type=int, required=True)
    parser.add_argument("--tp-size", type=int, required=True)
    parser.add_argument("--pp-size", type=int, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    main(args)
