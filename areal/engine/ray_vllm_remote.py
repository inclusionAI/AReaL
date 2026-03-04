import subprocess
import sys
import uuid
from typing import Any

import ray
from ray.runtime_env import RuntimeEnv
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from areal.api.cli_args import InferenceEngineConfig, vLLMConfig
from areal.engine.vllm_remote import RemotevLLMEngine, VLLMBackend, _copy_environ
from areal.infra import RemoteInfEngine
from areal.infra.utils.proc import kill_process_tree
from areal.infra.utils.ray import (
    create_resource_spec,
    get_placement_group_master_ip_and_port,
)
from areal.utils import logging

logger = logging.getLogger("RayVLLMRemote")


def _get_gpu(bundle):
    if "NPU" in bundle:
        return "NPU"
    elif "GPU" in bundle:
        return "GPU"

    return ""


def _get_resource_spec_and_n_gpu(bundle):
    cpu = bundle["CPU"]
    # already in bytes since it's from the bundle
    memory = bundle["memory"]

    # cannot be autodetected since this launcher is launched with 0 gpus
    # must read from bundle
    n_gpu = 0
    device = "CPU"
    if device := _get_gpu(bundle):
        n_gpu = bundle[device]

    return create_resource_spec(device, cpu, n_gpu, memory), n_gpu


@ray.remote
class vLLMMultinodeLauncher:
    def __init__(self):
        self.process = None

    def launch_server(self, server_args: dict[str, Any], headless):
        if headless:
            cmd = vLLMConfig.build_cmd_from_args_headless(server_args)
        else:
            cmd = vLLMConfig.build_cmd_from_args(server_args)

        _env = _copy_environ()

        self.process = subprocess.Popen(
            cmd, env=_env, stdout=sys.stdout, stderr=sys.stdout
        )

    def shutdown(self):
        logger.info("Received termination, killing vllm server process")
        if self.process and self.process.poll() is None:
            kill_process_tree(self.process.pid, graceful=True)

    def __ray_shutdown__(self):
        self.shutdown()


class RayVLLMBackend(VLLMBackend):
    """
    Same as VLLMBackend except uses Ray to launch the servers
    """

    def __init__(self):
        super().__init__()
        # save actors as strings instead of actor ref as actor ref is not serializable in ProcessPoolExecutor
        self.actor_names: list[str] = []
        # for dp
        self.dp_ip = ""
        self.dp_port = 0

    def launch_server(
        self, server_args: dict[str, Any]
    ) -> list[ray.actor.ActorHandle[vLLMMultinodeLauncher]]:
        pg = ray.util.get_current_placement_group()
        tp_size = server_args["tensor_parallel_size"]
        pp_size = server_args["pipeline_parallel_size"]
        dp_group_world_size = tp_size * pp_size
        dp_offset = 0
        is_head = True

        actors = []

        for i, bundle in enumerate(pg.bundle_specs):
            options, n_gpu = _get_resource_spec_and_n_gpu(bundle)
            current_args = server_args.copy()
            if is_head:
                self.dp_ip, self.dp_port = get_placement_group_master_ip_and_port(pg, i)
            logger.info(f"Launching actor {i}")

            # remove VISIBLE DEVICE envs as ray head had already set them
            # without removing them, vLLM cannot access devices
            # similarly with ray inherited env vars as they can cause scheduling issues
            _env = _copy_environ()
            new_env = {}
            for k, v in _env.items():
                if "VISIBLE_DEVICE" in k:
                    continue
                if "VISIBLE_CORE" in k:
                    continue
                if "RAY_" in k:
                    continue
                new_env[k] = v

            actor_name = str(uuid.uuid4())
            actor = vLLMMultinodeLauncher.options(
                **options,
                name=actor_name,
                lifetime="detached",
                runtime_env=RuntimeEnv(env_vars=new_env),
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_capture_child_tasks=True,
                    placement_group_bundle_index=i,
                ),
            ).remote()

            # cast to int as the args require integer values
            n_gpu = int(n_gpu)
            dp_local = n_gpu // dp_group_world_size
            current_args["data_parallel_size_local"] = dp_local
            current_args["data_parallel_address"] = self.dp_ip
            current_args["data_parallel_rpc_port"] = self.dp_port
            logger.info(f"Launching server {i}")
            if is_head:
                current_args["api_server_count"] = int(bundle["CPU"])
                actor.launch_server.remote(current_args, False)
                is_head = False
            else:
                current_args["data_parallel_start_rank"] = dp_offset
                actor.launch_server.remote(current_args, True)

            dp_offset += dp_local

            self.actor_names.append(actor_name)
            actors.append(actor)

        return actors


class RayRemotevLLMEngine(RemotevLLMEngine):
    """
    Same as RemotevLLMEngine except uses RayVLLMBackend for composition
    """

    def __init__(self, config: InferenceEngineConfig):
        self.config = config
        # Pure composition - create internal engine with vLLM backend
        self._engine = RemoteInfEngine(config, RayVLLMBackend())
        # lora already initialized when use_lora=true during init, by design, for vLLM
        self._engine.lora_initialized = config.use_lora
