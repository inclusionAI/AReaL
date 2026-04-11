#!/usr/bin/env python3
"""Torchrun entrypoint for awex Megatron <-> AReaL SGLang NCCL integration.

This script is intentionally generic over training-side parallel allocations and can
be launched by a controller pytest file with different strategies.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Mapping
from datetime import timedelta

import requests
import torch
import torch.distributed as dist

from areal.utils.network import find_free_ports


def _split_visible_devices(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [d.strip() for d in raw.split(",") if d.strip()]


def _early_pin_visible_device_from_local_rank() -> None:
    """Pin a single visible GPU before CUDA runtime is initialized.

    This runs at module import time and only takes effect in torchrun workers,
    where distributed env vars are present.
    """

    if not all(k in os.environ for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK")):
        return
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = _split_visible_devices(raw)
    if not devices:
        return
    os.environ.setdefault("AREAL_ORIG_CUDA_VISIBLE_DEVICES", raw or "")
    try:
        local_rank = int(os.environ["LOCAL_RANK"])
    except ValueError:
        return
    if local_rank < 0 or local_rank >= len(devices):
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = devices[local_rank]
    # Process now has singleton GPU visibility.
    os.environ["LOCAL_RANK"] = "0"
    os.environ["LOCAL_WORLD_SIZE"] = "1"
    os.environ["DEVICE"] = "0"


_early_pin_visible_device_from_local_rank()


def _log(rank: int, message: str) -> None:
    print(f"[awex-sglang-nccl][rank{rank}] {message}", flush=True)


def _probe_health(base_url: str, timeout_s: float = 2.0) -> tuple[bool, str]:
    try:
        resp = requests.get(f"{base_url}/health", timeout=timeout_s, json={})
        text = resp.text
        return resp.status_code == 200, f"{resp.status_code} {text}"
    except requests.exceptions.RequestException as exc:
        return False, f"request_error: {exc}"


def _sanitize_server_env(env: Mapping[str, str]) -> dict[str, str]:
    """Remove parent torchrun distributed env vars for child SGLang server.

    The child server runs its own distributed bootstrap and must not inherit the
    controller torchrun rendezvous variables.
    """

    blocked_prefixes = (
        "TORCHELASTIC_",
        "GROUP_",
        "ROLE_",
    )
    blocked_exact = {
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING",
        "PMI_RANK",
        "PMI_SIZE",
        "PMI_FD",
    }

    sanitized = dict(env)
    for key in list(sanitized.keys()):
        if key in blocked_exact or any(key.startswith(p) for p in blocked_prefixes):
            sanitized.pop(key, None)
    return sanitized


def _build_reader_server_env(
    parent_env: Mapping[str, str], host: str, dist_port: int, gpu_id: str
) -> dict[str, str]:
    env = _sanitize_server_env(parent_env)
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    env["DEVICE"] = "0"
    env["RANK"] = "0"
    env["WORLD_SIZE"] = "1"
    env["LOCAL_RANK"] = "0"
    env["LOCAL_WORLD_SIZE"] = "1"
    env["MASTER_ADDR"] = host
    env["MASTER_PORT"] = str(dist_port)
    env["TORCHELASTIC_USE_AGENT_STORE"] = "False"
    return env


def _configure_single_gpu_runtime_env(
    rank: int, world_size: int, local_rank: int, all_visible: list[str]
) -> str:
    del world_size
    selected = str(rank)
    if all_visible and 0 <= local_rank < len(all_visible):
        selected = all_visible[local_rank]
    os.environ["CUDA_VISIBLE_DEVICES"] = selected
    os.environ["LOCAL_RANK"] = "0"
    os.environ["LOCAL_WORLD_SIZE"] = "1"
    os.environ["DEVICE"] = "0"
    os.environ["TORCHELASTIC_USE_AGENT_STORE"] = "False"
    return selected


def _global_pg_init_method(host: str, port: int) -> str:
    return f"tcp://{host}:{port}"


def _write_result(path: str, ok: bool, message: str = "") -> None:
    if dist.get_rank() != 0:
        return
    payload = "Passed" if ok else f"Failed: {message}"
    with open(path, "w", encoding="utf-8") as f:
        f.write(payload)


def _run_with_timeout(name: str, timeout_s: int, fn, *args, **kwargs):
    """Run callable in thread with timeout and propagate exception."""

    outcome: dict[str, object] = {"done": False, "error": None}

    def _target():
        try:
            fn(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - runtime path
            outcome["error"] = exc
        finally:
            outcome["done"] = True

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout=timeout_s)
    if t.is_alive():
        raise TimeoutError(f"{name} timed out after {timeout_s}s")
    if outcome["error"] is not None:
        raise outcome["error"]  # type: ignore[misc]


def _destroy_dist_safely(rank: int) -> None:
    if not dist.is_initialized():
        return
    try:
        dist.destroy_process_group()
        _log(rank, "destroyed torch process group")
    except Exception as exc:  # pragma: no cover - cleanup path
        _log(rank, f"destroy_process_group failed: {exc}")


def _force_shutdown_on_signal(
    rank: int,
    signum: int,
    destroy_dist_fn,
    stop_server_and_meta_fn,
    exit_fn,
) -> None:
    """Best-effort emergency shutdown path for SIGINT/SIGTERM.

    Ensure distributed resources and child processes are torn down before
    terminating the current process to prevent lingering NCCL heartbeat logs.
    """

    _log(rank, f"received signal {signum}; forcing shutdown")
    try:
        destroy_dist_fn(rank)
    finally:
        stop_server_and_meta_fn()
    exit_fn(128 + signum)


def _visible_devices() -> list[str]:
    original = _split_visible_devices(os.environ.get("AREAL_ORIG_CUDA_VISIBLE_DEVICES"))
    if original:
        return original
    env_name = "CUDA_VISIBLE_DEVICES"
    if env_name in os.environ and os.environ[env_name].strip():
        return [d.strip() for d in os.environ[env_name].split(",") if d.strip()]
    if torch.cuda.is_available():
        return [str(i) for i in range(torch.cuda.device_count())]
    return []


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
    _validate_allocation(world_size, args.dp_size, args.tp_size, args.pp_size)

    _log(
        rank,
        "checkpoint/bootstrap: debug env "
        f"NCCL_CUMEM_ENABLE={os.environ.get('NCCL_CUMEM_ENABLE')} "
        f"NCCL_NVLS_ENABLE={os.environ.get('NCCL_NVLS_ENABLE')} "
        f"NCCL_DEBUG={os.environ.get('NCCL_DEBUG')} "
        f"NCCL_DEBUG_SUBSYS={os.environ.get('NCCL_DEBUG_SUBSYS')} "
        f"TORCH_DISTRIBUTED_DEBUG={os.environ.get('TORCH_DISTRIBUTED_DEBUG')} "
        f"TORCH_NCCL_ENABLE_MONITORING={os.environ.get('TORCH_NCCL_ENABLE_MONITORING')} "
        f"TORCHELASTIC_USE_AGENT_STORE={os.environ.get('TORCHELASTIC_USE_AGENT_STORE')}",
    )

    from awex.engine.mcore import MegatronEngine
    from awex.meta.meta_server import start_meta_server, stop_meta_server
    from awex.tests.test_utils import megatron_model_from_hf

    from areal.infra.utils.proc import kill_process_tree
    from areal.utils import network

    _log(
        rank,
        "checkpoint/bootstrap: cuda runtime view "
        f"device_count={torch.cuda.device_count()} current_device={torch.cuda.current_device()} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}",
    )

    dist.init_process_group(
        "nccl",
        timeout=timedelta(seconds=args.dist_timeout),
    )
    _log(rank, "checkpoint/global-pg: initialized")

    server_proc: subprocess.Popen | None = None
    meta_started = False

    def _stop_server_and_meta() -> None:
        if rank != 0:
            return
        if server_proc is not None:
            kill_process_tree(server_proc.pid, graceful=True)
            if server_proc.poll() is None:
                kill_process_tree(server_proc.pid, graceful=False)
        if meta_started:
            stop_meta_server()

    shutdown_once = {"done": False}

    def _on_terminate(signum, _frame):
        if shutdown_once["done"]:
            return
        shutdown_once["done"] = True
        _force_shutdown_on_signal(
            rank=rank,
            signum=signum,
            destroy_dist_fn=_destroy_dist_safely,
            stop_server_and_meta_fn=_stop_server_and_meta,
            exit_fn=os._exit,
        )

    signal.signal(signal.SIGINT, _on_terminate)
    signal.signal(signal.SIGTERM, _on_terminate)

    try:
        if rank == 0:
            _log(rank, "starting awex meta server")
            meta_ip, meta_port = start_meta_server()
            meta_started = True
            os.environ["AWEX_META_ADDR_BCAST"] = f"{meta_ip}:{meta_port}"
            host = network.gethostip()
            sglang_port, sglang_dist_port = find_free_ports(2)
            os.environ["AWEX_SGLANG_HOST_BCAST"] = host
            os.environ["AWEX_SGLANG_PORT_BCAST"] = str(sglang_port)

            from areal.api.cli_args import SGLangConfig

            cmd = SGLangConfig.build_cmd(
                sglang_config=SGLangConfig(
                    skip_tokenizer_init=True,
                    model_path=args.model_path,
                    mem_fraction_static=0.15,
                    launch_server_module="areal.engine.sglang_ext.areal_sglang_server",
                ),
                host=host,
                port=sglang_port,
                tp_size=1,
                base_gpu_id=0,
                dist_init_addr=f"{host}:{sglang_dist_port}",
            )
            gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
            server_env = _build_reader_server_env(
                parent_env=os.environ,
                host=host,
                dist_port=sglang_dist_port,
                gpu_id=gpu_id or "0",
            )
            # Prefer a non-overlapping GPU when available; otherwise colocate.
            server_proc = subprocess.Popen(
                cmd,
                env=server_env,
                stdout=sys.stdout,
                stderr=sys.stdout,
            )
            _log(rank, f"sglang process command issued: {' '.join(cmd)}")

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
        _log(rank, f"received endpoints: meta={meta_addr} sglang={base_url}")

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
                        "enable_debug_mode": False,
                        "debug_mode_config": {
                            "enable_nccl_debug_mode": True,
                            "raise_on_validation_fail": False,
                        },
                        "nnodes": 1,
                        "node_rank": 0,
                    },
                    timeout=max(30, args.rpc_timeout),
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
                    timeout=max(30, args.rpc_timeout),
                )
                if resp.status_code == 200 and resp.json().get("success"):
                    update_result["ok"] = True
                else:
                    update_result["error"] = (
                        f"update failed: {resp.status_code} {resp.text}"
                    )
            except Exception as exc:  # pragma: no cover
                update_result["error"] = str(exc)

        health_gate = [False, ""]
        if rank == 0:
            deadline = time.time() + args.health_timeout
            _log(
                rank,
                f"waiting for sglang health at {base_url} (timeout={args.health_timeout}s)",
            )
            last_detail = ""
            last_log_ts = 0.0
            while time.time() < deadline:
                if server_proc is not None and server_proc.poll() is not None:
                    raise RuntimeError(
                        f"AReaL SGLang server exited early with code {server_proc.returncode}"
                    )
                ok, detail = _probe_health(base_url, timeout_s=2.0)
                last_detail = detail
                now = time.time()
                if now - last_log_ts >= 10:
                    elapsed = int(now - (deadline - args.health_timeout))
                    _log(rank, f"health probe pending (elapsed={elapsed}s): {detail}")
                    last_log_ts = now
                if detail.startswith("500 "):
                    health_gate[1] = (
                        f"AReaL SGLang server reported fatal health error: {detail}"
                    )
                    break
                if ok:
                    _log(rank, "sglang server is healthy")
                    health_gate[0] = True
                    break
                time.sleep(1)
            else:
                health_gate[1] = (
                    "AReaL SGLang server did not become healthy within "
                    f"{args.health_timeout}s; last health probe: {last_detail}"
                )

        dist.broadcast_object_list(health_gate, src=0)
        health_ok, health_err = bool(health_gate[0]), str(health_gate[1])
        if not health_ok:
            raise RuntimeError(health_err)

        if rank == 0:
            init_thread = threading.Thread(target=_call_awex_init, daemon=True)
            _log(rank, "checkpoint/awex-init: dispatching init request")
            init_thread.start()
        else:
            init_thread = None

        _log(rank, "initializing megatron parallel + model")
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
        # Disable torch agent store for torchrun,
        # otherwise the process group won't be properly initialized.
        os.environ["TORCHELASTIC_USE_AGENT_STORE"] = str(False)
        megatron_engine.initialize()
        _log(rank, "megatron engine initialized")

        if rank == 0 and init_thread is not None:
            init_thread.join(timeout=args.rpc_timeout)
            if init_thread.is_alive() and init_result["error"] is None:
                init_result["error"] = f"awex init timed out after {args.rpc_timeout}s"
            init_gate = [bool(init_result["ok"]), str(init_result["error"] or "")]
            _log(rank, f"checkpoint/awex-init: result ok={init_gate[0]}")
            if init_gate[0]:
                _log(rank, "awex reader init done")
        else:
            init_gate = [False, ""]

        dist.broadcast_object_list(init_gate, src=0)
        if not bool(init_gate[0]):
            raise RuntimeError(f"Reader init failed: {init_gate[1]}")

        dist.barrier()

        step_id = 1
        if rank == 0:
            update_thread = threading.Thread(
                target=_call_awex_update, kwargs={"step_id": step_id}, daemon=True
            )
            _log(rank, f"checkpoint/awex-update: dispatching step_id={step_id}")
            update_thread.start()
            # Give update path a short grace window to surface immediate failures
            # (e.g., awex reader init/update validation errors) before writer
            # enters long infer_conf wait paths.
            prewrite_deadline = time.time() + 5
            while time.time() < prewrite_deadline:
                if update_result["error"]:
                    raise RuntimeError(
                        f"Reader update failed early: {update_result['error']}"
                    )
                if update_result["ok"]:
                    break
                if not update_thread.is_alive():
                    break
                time.sleep(0.1)
            if update_result["error"]:
                raise RuntimeError(
                    f"Reader update failed early: {update_result['error']}"
                )
        else:
            update_thread = None

        _log(rank, "writing weights from megatron engine")
        megatron_engine.set_global_step(step_id)
        _log(rank, f"checkpoint/write: global_step={step_id} entering write_weights")
        _run_with_timeout(
            "megatron_engine.write_weights",
            max(30, args.rpc_timeout),
            megatron_engine.write_weights,
        )
        _log(rank, "weights write completed")

        if rank == 0 and update_thread is not None:
            update_thread.join(timeout=args.rpc_timeout)
            if update_thread.is_alive() and update_result["error"] is None:
                update_result["error"] = (
                    f"awex update timed out after {args.rpc_timeout}s"
                )
            update_gate = [
                bool(update_result["ok"]),
                str(update_result["error"] or ""),
            ]
            _log(rank, f"checkpoint/awex-update: result ok={update_gate[0]}")
            if update_gate[0]:
                _log(rank, "awex reader update done")
        else:
            update_gate = [False, ""]

        dist.broadcast_object_list(update_gate, src=0)
        if not bool(update_gate[0]):
            raise RuntimeError(f"Reader update failed: {update_gate[1]}")

        dist.barrier()
        _write_result(args.output, True)
        _log(rank, "test run passed")
    except Exception as exc:
        tb = traceback.format_exc(limit=5)
        _log(rank, f"FAILED: {exc}\n{tb}")
        _write_result(args.output, False, f"{exc}\n{tb}")
        _destroy_dist_safely(rank)
        _stop_server_and_meta()
        raise
    finally:
        _destroy_dist_safely(rank)
        _stop_server_and_meta()


def _amend_init_environs():
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    all_visible = _visible_devices()
    _configure_single_gpu_runtime_env(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        all_visible=all_visible,
    )
    # In virtualized/containerized environments NCCL P2P can fail with
    # ncclUnhandledCudaError/invalid argument on cross-process exchange.
    # Prefer a more portable transport path for this integration test.
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_CUMEM_ENABLE", "0")
    os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
    os.environ.setdefault("NCCL_DEBUG", "INFO")
    os.environ.setdefault("NCCL_DEBUG_SUBSYS", "INIT,COLL,GRAPH,TUNING")
    os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "DETAIL")
    os.environ.setdefault("TORCH_NCCL_ENABLE_MONITORING", "0")
    from awex.util import device as device_util

    device_util.set_device(0)


if __name__ == "__main__":
    _amend_init_environs()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dp-size", type=int, required=True)
    parser.add_argument("--tp-size", type=int, required=True)
    parser.add_argument("--pp-size", type=int, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--health-timeout", type=int, default=300)
    parser.add_argument("--rpc-timeout", type=int, default=300)
    parser.add_argument("--dist-timeout", type=int, default=600)
    args = parser.parse_args()
    main(args)
