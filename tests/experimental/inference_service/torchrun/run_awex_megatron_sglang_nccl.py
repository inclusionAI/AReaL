#!/usr/bin/env python3
"""Torchrun entrypoint for awex Megatron <-> AReaL SGLang NCCL integration.

This script is intentionally generic over training-side parallel allocations and can
be launched by a controller pytest file with different strategies.
"""

from __future__ import annotations

import argparse
import os
import signal
import threading
import time
import traceback
from datetime import timedelta

import requests
import torch
import torch.distributed as dist


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


def _log(rank: int, message: str) -> None:
    print(f"[awex-sglang-nccl][rank{rank}] {message}", flush=True)


def _probe_health(base_url: str, timeout_s: float = 2.0) -> tuple[bool, str]:
    try:
        resp = requests.get(f"{base_url}/health", timeout=timeout_s, json={})
        text = resp.text
        return resp.status_code == 200, f"{resp.status_code} {text}"
    except requests.exceptions.RequestException as exc:
        return False, f"request_error: {exc}"


def _write_result(path: str, ok: bool, message: str = "") -> None:
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    payload = "Passed" if ok else f"Failed: {message}"
    with open(path, "w", encoding="utf-8") as f:
        f.write(payload)


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


def _sanitize_server_env(parent_env: dict[str, str]) -> dict[str, str]:
    """Drop torchrun launcher vars before spawning isolated reader workers."""

    env = dict(parent_env)
    for key in (
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "TORCHELASTIC_USE_AGENT_STORE",
    ):
        env.pop(key, None)
    return env


def _global_pg_init_method(host: str, dist_port: int) -> str:
    return f"tcp://{host}:{dist_port}"


def _build_reader_server_env(
    parent_env: dict[str, str],
    host: str,
    dist_port: int,
    gpu_id: str,
) -> dict[str, str]:
    """Build single-process dist env for an isolated reader server process."""

    env = _sanitize_server_env(parent_env)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["DEVICE"] = "0"
    env["RANK"] = "0"
    env["WORLD_SIZE"] = "1"
    env["LOCAL_RANK"] = "0"
    env["LOCAL_WORLD_SIZE"] = "1"
    env["MASTER_ADDR"] = str(host)
    env["MASTER_PORT"] = str(dist_port)
    env["TORCHELASTIC_USE_AGENT_STORE"] = str(False)
    return env


def _configure_single_gpu_runtime_env(
    rank: int,
    world_size: int,
    local_rank: int,
    all_visible: list[str],
) -> str:
    """Normalize per-process runtime env to singleton CUDA visibility."""

    if local_rank < 0 or local_rank >= len(all_visible):
        raise ValueError(
            f"Invalid local_rank={local_rank} for visible devices={all_visible}"
        )
    selected = all_visible[local_rank]
    os.environ.setdefault("AREAL_ORIG_CUDA_VISIBLE_DEVICES", ",".join(all_visible))
    os.environ["CUDA_VISIBLE_DEVICES"] = selected
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = "0"
    os.environ["LOCAL_WORLD_SIZE"] = "1"
    os.environ["DEVICE"] = "0"
    os.environ["TORCHELASTIC_USE_AGENT_STORE"] = str(False)
    return selected


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


def _validate_awex_tp_contract(train_tp_size: int, infer_tp_size: int) -> None:
    """Validate AWEX tp-resharding contract early with a clear error.

    AWEX metadata alignment enforces: infer_tp >= train_tp and infer_tp % train_tp == 0.
    """

    if infer_tp_size < train_tp_size or infer_tp_size % train_tp_size != 0:
        raise ValueError(
            "Unsupported AWEX TP topology: "
            f"train_tp_size={train_tp_size}, infer_tp_size={infer_tp_size}. "
            "AWEX requires infer_tp_size >= train_tp_size and "
            "infer_tp_size % train_tp_size == 0."
        )


def main(args: argparse.Namespace) -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    _validate_allocation(world_size, args.dp_size, args.tp_size, args.pp_size)
    _validate_awex_tp_contract(
        train_tp_size=args.tp_size,
        infer_tp_size=args.infer_tp_size,
    )

    _log(
        rank,
        "checkpoint/bootstrap: debug env "
        f"NCCL_SHM_DISABLE={os.environ.get('NCCL_SHM_DISABLE')} "
        f"NCCL_IB_DISABLE={os.environ.get('NCCL_IB_DISABLE')} "
        f"NCCL_CUMEM_ENABLE={os.environ.get('NCCL_CUMEM_ENABLE')} "
        f"NCCL_NVLS_ENABLE={os.environ.get('NCCL_NVLS_ENABLE')} "
        f"NCCL_DEBUG={os.environ.get('NCCL_DEBUG')} "
        f"NCCL_DEBUG_SUBSYS={os.environ.get('NCCL_DEBUG_SUBSYS')} "
        f"TORCH_DISTRIBUTED_DEBUG={os.environ.get('TORCH_DISTRIBUTED_DEBUG')} "
        f"TORCH_NCCL_ENABLE_MONITORING={os.environ.get('TORCH_NCCL_ENABLE_MONITORING')} "
        f"TORCHELASTIC_USE_AGENT_STORE={os.environ.get('TORCHELASTIC_USE_AGENT_STORE')}",
    )

    from awex.engine.mcore import MegatronEngine as AwexMegatronEngine
    from awex.meta.meta_server import start_meta_server, stop_meta_server
    from awex.tests.test_utils import megatron_model_from_hf

    class MegatronEngine(AwexMegatronEngine):
        def initialize(self):
            super().initialize()
            # HACK:
            self.weights_exchange_writer.asystem_train_config = (
                self.weights_exchange_writer.config
            )
            # FIXME: awex doesn't work on a single colocated GPU

        def resume_memory_occupation(self, tags=None):
            pass

        def release_grad_memory(self, empty_cache=True):
            pass

    _log(
        rank,
        "checkpoint/bootstrap: cuda runtime view "
        f"device_count={torch.cuda.device_count()} current_device={torch.cuda.current_device()} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}",
    )

    dist.init_process_group("nccl", timeout=timedelta(seconds=args.dist_timeout))
    _log(rank, "checkpoint/global-pg: initialized")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    meta_started = False

    def _stop_server_and_meta() -> None:
        if rank != 0:
            return
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
            base_urls = [
                u.strip() for u in args.awex_server_urls.split(",") if u.strip()
            ]
            if not base_urls:
                raise ValueError("--awex-server-urls must include at least one URL")
            shared = [f"{meta_ip}:{meta_port}", base_urls]
        else:
            shared = ["", []]

        # Broadcast meta address + externally managed awex server endpoint.
        dist.broadcast_object_list(shared, src=0)
        meta_addr, base_urls = shared
        if not isinstance(base_urls, list) or not base_urls:
            raise RuntimeError("No awex server URLs broadcast from rank0")
        _log(
            rank,
            f"received endpoints: meta={meta_addr} sglang_count={len(base_urls)} first={base_urls[0]}",
        )

        init_result: dict[str, object] = {"ok": False, "error": None}
        update_result: dict[str, object] = {"ok": False, "error": None}

        def _call_awex_init() -> None:
            try:
                num_engines = len(base_urls)
                for engine_rank, base_url in enumerate(base_urls):
                    resp = requests.post(
                        f"{base_url}/areal_awex_init",
                        json={
                            "meta_server_addr": meta_addr,
                            "engine_rank": engine_rank,
                            "num_engines": num_engines,
                            "comm_backend": "nccl",
                            "enable_debug_mode": False,
                            "debug_mode_config": {
                                "enable_nccl_debug_mode": True,
                                "raise_on_validation_fail": False,
                            },
                            "enable_colocate_mode": False,
                            "nnodes": 1,
                            "node_rank": 0,
                        },
                        timeout=max(30, args.rpc_timeout),
                    )
                    if not (resp.status_code == 200 and resp.json().get("success")):
                        init_result["error"] = (
                            f"init failed for engine_rank={engine_rank} url={base_url}: "
                            f"{resp.status_code} {resp.text}"
                        )
                        return
                init_result["ok"] = True
            except Exception as exc:  # pragma: no cover
                init_result["error"] = str(exc)

        def _call_awex_update(step_id: int) -> None:
            try:
                for engine_rank, base_url in enumerate(base_urls):
                    resp = requests.post(
                        f"{base_url}/areal_awex_update",
                        json={"step_id": step_id, "kwargs": {}},
                        timeout=max(30, args.rpc_timeout),
                    )
                    if not (resp.status_code == 200 and resp.json().get("success")):
                        update_result["error"] = (
                            f"update failed for engine_rank={engine_rank} url={base_url}: "
                            f"{resp.status_code} {resp.text}"
                        )
                        return
                update_result["ok"] = True
            except Exception as exc:  # pragma: no cover
                update_result["error"] = str(exc)

        health_gate = [False, ""]
        if rank == 0:
            deadline = time.time() + args.health_timeout
            _log(
                rank,
                f"waiting for all sglang workers health (timeout={args.health_timeout}s)",
            )
            last_detail = ""
            last_log_ts = 0.0
            while time.time() < deadline:
                all_ok = True
                details: list[str] = []
                for idx, base_url in enumerate(base_urls):
                    ok, detail = _probe_health(base_url, timeout_s=2.0)
                    details.append(f"{idx}:{detail}")
                    if not ok:
                        all_ok = False
                last_detail = " | ".join(details)
                now = time.time()
                if now - last_log_ts >= 10:
                    elapsed = int(now - (deadline - args.health_timeout))
                    _log(
                        rank,
                        f"health probe pending (elapsed={elapsed}s): {last_detail}",
                    )
                    last_log_ts = now
                if any(d.split(":", 1)[1].startswith("500 ") for d in details):
                    health_gate[1] = (
                        "AReaL SGLang server reported fatal health error: "
                        f"{last_detail}"
                    )
                    break
                if all_ok:
                    _log(rank, "all sglang workers are healthy")
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
            "enable_colocate_mode": False,
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
        else:
            update_thread = None

        _log(rank, "writing weights from megatron engine")
        megatron_engine.set_global_step(step_id)
        _log(rank, f"checkpoint/write: global_step={step_id} entering write_weights")
        megatron_engine.write_weights()
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dp-size", type=int, required=True)
    parser.add_argument("--tp-size", type=int, required=True)
    parser.add_argument("--pp-size", type=int, required=True)
    parser.add_argument("--infer-tp-size", type=int, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument(
        "--awex-server-urls",
        "--awex-server-url",
        dest="awex_server_urls",
        type=str,
        required=True,
        help="Comma-separated inference worker base URLs.",
    )
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--health-timeout", type=int, default=300)
    parser.add_argument("--rpc-timeout", type=int, default=300)
    parser.add_argument("--dist-timeout", type=int, default=600)
    args = parser.parse_args()
    main(args)
