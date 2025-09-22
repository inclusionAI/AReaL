import os
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import requests

from areal.api.cli_args import (
    ClusterSpecConfig,
    NameResolveConfig,
    parse_cli_args,
    to_structured_cfg,
    vLLMConfig,
)
from areal.api.io_struct import AllocationMode
from areal.platforms import is_npu_available
from areal.utils.launcher import TRITON_CACHE_PATH
from areal.utils.network import find_free_ports, gethostip
from realhf.base import logging, name_resolve, names

logger = logging.getLogger("vLLMServer Wrapper")


def launch_server_cmd(command: str) -> subprocess.Popen:
    """
    Execute a shell command and return its process handle.
    """
    # Replace newline continuations and split the command string.
    command = command.replace("\\\n", " ").replace("\\", " ")
    parts = command.split()
    _env = os.environ.copy()
    # To avoid DirectoryNotEmpty error caused by triton
    triton_cache_path = _env.get("TRITON_CACHE_PATH", TRITON_CACHE_PATH)
    unique_triton_cache_path = os.path.join(triton_cache_path, str(uuid.uuid4()))
    _env["TRITON_CACHE_PATH"] = unique_triton_cache_path
    return subprocess.Popen(
        parts,
        text=True,
        env=_env,
        stdout=sys.stdout,
        stderr=subprocess.STDOUT,
    )


def wait_for_server(base_url: str, timeout: Optional[int] = None) -> None:
    """Wait for the server to be ready by polling the /v1/models endpoint.

    Args:
        base_url: The base URL of the server
        timeout: Maximum time to wait in seconds. None means wait forever.
    """
    start_time = time.time()
    while True:
        try:
            response = requests.get(
                f"{base_url}/v1/models",
                headers={"Authorization": "Bearer None"},
            )
            if response.status_code == 200:
                time.sleep(5)
                break

            if timeout and time.time() - start_time > timeout:
                raise TimeoutError("Server did not become ready within timeout period")
        except requests.exceptions.RequestException:
            time.sleep(1)


class vLLMServerWrapper:
    def __init__(
        self,
        experiment_name: str,
        trial_name: str,
        vllm_config: vLLMConfig,
        allocation_mode: AllocationMode,
        n_gpus_per_node: int,
    ):
        self.experiment_name = experiment_name
        self.trial_name = trial_name
        self.config = vllm_config
        self.allocation_mode = allocation_mode
        self.server_process = None
        self.n_gpus_per_node = n_gpus_per_node

    def run(self):
        gpus_per_server = self.allocation_mode.gen_instance_size
        if gpus_per_server > self.n_gpus_per_node:
            raise NotImplementedError("Cross-node vllm is not supported")

        n_servers_per_node = max(1, self.n_gpus_per_node // gpus_per_server)
        if is_npu_available:
            if "ASCEND_RT_VISIBLE_DEVICES" in os.environ:
                visible = os.getenv("ASCEND_RT_VISIBLE_DEVICES").split(",")
                ordered = ",".join(sorted(visible, key=int))
                os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ordered
                n_visible_devices = len(visible)
                n_servers_per_proc = max(1, n_visible_devices // gpus_per_server)
                server_idx_offset = int(visible[0]) // gpus_per_server
            else:
                n_visible_devices = self.n_gpus_per_node
                n_servers_per_proc = n_servers_per_node
                server_idx_offset = 0
        else:
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                visible = os.getenv("CUDA_VISIBLE_DEVICES").split(",")
                n_visible_devices = len(visible)
                n_servers_per_proc = max(1, n_visible_devices // gpus_per_server)
                server_idx_offset = int(visible[0]) // gpus_per_server
            else:
                n_visible_devices = self.n_gpus_per_node
                n_servers_per_proc = n_servers_per_node
                server_idx_offset = 0

        # Separate ports used by each server in the same node
        # ports range (10000, 50000)
        ports_per_server = 40000 // n_servers_per_node
        launch_server_args = []
        server_addresses = []
        for server_local_idx in range(
            server_idx_offset, server_idx_offset + n_servers_per_proc
        ):
            port_range = (
                server_local_idx * ports_per_server + 10000,
                (server_local_idx + 1) * ports_per_server + 10000,
            )
            server_port, dist_init_port = find_free_ports(2, port_range)

            dist_init_addr = f"localhost:{dist_init_port}"
            host_ip = gethostip()

            (server_local_idx - server_idx_offset) * gpus_per_server
            cmd = vLLMConfig.build_cmd(
                self.config,
                tp_size=self.allocation_mode.gen.tp_size,
                host=host_ip,
                port=server_port,
                dist_init_addr=dist_init_addr,
            )
            launch_server_args.append((cmd, host_ip, server_port))
            server_addresses.append(f"http://{host_ip}:{server_port}")

        with ThreadPoolExecutor(max_workers=n_servers_per_proc) as executor:
            server_processes = executor.map(
                lambda args: self.launch_one_server(*args), launch_server_args
            )

        while True:
            all_alive = True
            for i, process in enumerate(server_processes):
                return_code = process.poll()
                if return_code is not None:
                    logger.info(
                        f"vllm server {server_addresses[i]} exits, returncode={return_code}"
                    )
                    all_alive = False
                    break

            if not all_alive:
                for i, process in enumerate(server_processes):
                    if process.poll() is None:
                        process.terminate()
                        process.wait()
                        logger.info(
                            f"vllm server process{server_addresses[i]} terminated."
                        )

            time.sleep(1)

    def launch_one_server(self, cmd, host_ip, server_port):
        server_process = launch_server_cmd(cmd)
        wait_for_server(f"http://{host_ip}:{server_port}")
        name = names.gen_servers(self.experiment_name, self.trial_name)
        name_resolve.add_subentry(name, f"{host_ip}:{server_port}")
        logger.info(f"vllm server launched at: http://{host_ip}:{server_port}")
        return server_process


def main(argv):
    config, _ = parse_cli_args(argv)
    config.vllm = to_structured_cfg(config.vllm, vLLMConfig)
    config.cluster = to_structured_cfg(config.cluster, ClusterSpecConfig)
    config.cluster.name_resolve = to_structured_cfg(
        config.cluster.name_resolve, NameResolveConfig
    )
    name_resolve.reconfigure(config.cluster.name_resolve)

    allocation_mode = config.allocation_mode
    allocation_mode = AllocationMode.from_str(allocation_mode)
    assert allocation_mode.gen_backend == "vllm"

    vllm_server = vLLMServerWrapper(
        config.experiment_name,
        config.trial_name,
        config.vllm,
        allocation_mode,
        n_gpus_per_node=config.cluster.n_gpus_per_node,
    )
    vllm_server.run()


if __name__ == "__main__":
    main(sys.argv[1:])
