import importlib.util
import sys
import threading
from pathlib import Path

import ray
from fastapi import FastAPI, Request
from omegaconf import OmegaConf
from uvicorn import Config, Server

import areal.utils.logging as logging
from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import (
    ClusterSpecConfig,
    LauncherConfig,
    RecoverConfig,
    ScalingConfig,
    to_structured_cfg,
    vLLMConfig,
)
from areal.platforms import is_npu_available
from areal.utils import name_resolve
from areal.utils.launcher import get_env_vars, wait_llm_server_addrs
from areal.utils.name_resolve import NameEntryNotFoundError

logger = logging.getLogger("ScaleUpVLLM")
DEFAULT_MAIN_FUNC_NAME = "main"


def run_func(file_path: str, func_name: str, argv: list[str]):
    """
    Import module by path and invoke the named function with a single `argv` list.
    This matches vllm_server.main(argv) which expects sys.argv[2:]-style args.
    """
    module_name = file_path.replace("/", "_").replace(".", "_")
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    func = getattr(module, func_name)
    return func(argv)


def scale_up_vllm(
    cfg, config_path: str, n_new_servers: int, expected: int, vllm_entry_point: str
):
    # Make sub-configs structured like original launcher
    cfg.launcher = to_structured_cfg(cfg.launcher, LauncherConfig)
    cfg.cluster = to_structured_cfg(cfg.cluster, ClusterSpecConfig)
    cfg.recover = to_structured_cfg(cfg.recover, RecoverConfig)
    cfg.vllm = to_structured_cfg(cfg.vllm, vLLMConfig)
    experiment_name = cfg.experiment_name
    trial_name = cfg.trial_name

    allocation_mode = AllocationMode.from_str(cfg.allocation_mode)
    vllm_tp_size = allocation_mode.gen.tp_size
    n_existing_servers = expected - n_new_servers

    cpus_per_gpu = cfg.launcher.inference_server_cpus_per_gpu
    mem_per_gpu = cfg.launcher.inference_server_mem_per_gpu

    # Submit new servers
    remote_runner = None  # we’ll bind ray.remote per device type
    futures = []
    for i in range(n_new_servers):
        argv = ["--config", config_path]
        # Env vars same as original launcher for inference servers
        env_vars = get_env_vars(
            cfg.cluster.cluster_name, cfg.launcher.inference_server_env_vars
        )

        if is_npu_available:
            remote_runner = ray.remote(
                num_cpus=cpus_per_gpu * vllm_tp_size,
                resources={"NPU": vllm_tp_size},
                memory=mem_per_gpu * vllm_tp_size * 1024 * 1024,  # bytes
                runtime_env={"env_vars": env_vars},
            )(run_func)
        else:
            remote_runner = ray.remote(
                num_cpus=cpus_per_gpu * vllm_tp_size,
                num_gpus=vllm_tp_size,
                memory=mem_per_gpu * vllm_tp_size * 1024 * 1024,  # bytes
                runtime_env={"env_vars": env_vars},
            )(run_func)

        fut = remote_runner.remote(vllm_entry_point, DEFAULT_MAIN_FUNC_NAME, argv)
        futures.append(fut)

        try:
            ray.get(fut, timeout=5.0)
        except ray.exceptions.GetTimeoutError:
            pass
        except ray.exceptions.RayTaskError as e:
            logger.info(f"[ERROR] server {n_existing_servers + i} crashed immediately:")
            logger.info(e)
            raise

    # Wait until ALL (old + new) servers are registered
    total_expected = expected
    vllm_addrs = wait_llm_server_addrs(
        experiment_name,
        trial_name,
        total_expected,
    )

    logger.info("\n[Scale-Up Completed]")
    logger.info(f"Total servers expected: {len(vllm_addrs)}")


app = FastAPI()
shared_state = {
    "cfg": None,
    "config_path": None,
    "num_rollout": None,
    "vllm_entry_point": None,
}
shared_state_lock = threading.Lock()


@app.post("/scale_up")
async def http_scale_up(request: Request):
    """
    Scaling controller endpoint.
    Example usage:
      curl -X POST localhost:8899/scale_up \
        -H "Content-Type: application/json" \
        -d '{"scaled_k": 1}'
    """
    body = await request.json()
    scaled_k = int(body.get("scaled_k", 1))

    with shared_state_lock:
        cfg = shared_state["cfg"]
        config_path = shared_state["config_path"]
        num_rollout = shared_state["num_rollout"]
        vllm_entry_point = shared_state["vllm_entry_point"]

        # More complete initialization check
        if (
            cfg is None
            or config_path is None
            or num_rollout is None
            or vllm_entry_point is None
        ):
            return {"status": "error", "msg": "Scale server not initialized yet"}

        new_total = num_rollout + scaled_k
        shared_state["num_rollout"] = new_total

    try:
        logger.info(f"[HTTP] Received manual scale-up request: {scaled_k}")
        name_resolve.add("scale_up_request", {"scaled_k": int(scaled_k)}, replace=True)

        scale_up_vllm(
            cfg,
            config_path,
            scaled_k,
            new_total,
            vllm_entry_point,
        )
        try:
            name_resolve.delete("scale_up_done")
        except NameEntryNotFoundError:
            pass

        name_resolve.add("scale_up_done", {"done": 1})
        logger.info(f"[HTTP] Scale-up done. Total rollout={new_total}")
        return {
            "status": "ok",
            "scaled_k": scaled_k,
            "new_total": new_total,
        }
    except Exception as e:
        logger.error(f"[HTTP] Scale-up failed: {e}")
        return {"status": "error", "msg": str(e)}


def run_http_server(port: int):
    """Run FastAPI server in background thread (non-blocking)."""
    config = Config(app, host="0.0.0.0", port=port, log_level="info")
    server = Server(config)

    def _serve():
        logger.info(f"[HTTP] Starting scaling controller server on port {port}")
        server.run()

    t = threading.Thread(target=_serve, daemon=False)
    t.start()
    logger.info("[HTTP] Scaling controller server started in background.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        logger.info("Usage: python scaling_controller <config.yaml> ")
        sys.exit(1)

    config_path = sys.argv[1]
    cfg = OmegaConf.load(config_path)
    name_resolve.reconfigure(cfg.cluster.name_resolve)
    experiment_name = cfg.experiment_name
    trial_name = cfg.trial_name

    allocation_mode = AllocationMode.from_str(cfg.allocation_mode)
    num_rollout = allocation_mode.gen.dp_size

    # Remove all the keys related to scaling before start the experiment
    try:
        name_resolve.delete("scale_up_request")
    except NameEntryNotFoundError:
        pass

    try:
        name_resolve.delete("scale_up_done")
    except NameEntryNotFoundError:
        pass

    # Init ray and connect it to existing cluster
    ray.init(address="auto", namespace=f"{experiment_name}_{trial_name}")

    # Get port for scale up
    cfg.scaling = to_structured_cfg(cfg.scaling, ScalingConfig)
    port = cfg.scaling.scaling_controller_port

    # Resolve vLLM entry point
    vllm_entry_point = str(Path(__file__).resolve().parent.parent / "vllm_server.py")

    # Initialize shared_state atomically before starting HTTP server
    with shared_state_lock:
        shared_state["cfg"] = cfg
        shared_state["config_path"] = config_path
        shared_state["num_rollout"] = num_rollout
        shared_state["vllm_entry_point"] = vllm_entry_point

    logger.info(f"[HTTP] num_rollout initialized to {num_rollout}")

    # Run http for scale-up (after shared_state is fully initialized)
    run_http_server(port)
