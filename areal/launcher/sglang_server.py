import os
import subprocess
import sys
import time
import uuid
from typing import Optional

import requests

from areal.api.cli_args import (
    ClusterSpecConfig,
    NameResolveConfig,
    SGLangConfig,
    parse_cli_args,
    to_structured_cfg,
)
from areal.api.io_struct import AllocationMode, AllocationType
from areal.utils.launcher import TRITON_CACHE_PATH
from areal.utils.network import find_free_ports, gethostip
from realhf.base import logging, name_resolve, names

logger = logging.getLogger("SGLangServer Wrapper")


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


class SGLangServerWrapper:
    def __init__(
        self,
        experiment_name: str,
        trial_name: str,
        sglang_config: SGLangConfig,
        tp_size: int,
        n_gpus_per_node: int,
    ):
        self.experiment_name = experiment_name
        self.trial_name = trial_name
        self.config = sglang_config
        self.tp_size = tp_size
        self.server_process = None
        self.n_gpus_per_node = n_gpus_per_node

    def run(self):
        gpus_per_server = len(os.getenv("CUDA_VISIBLE_DEVICES").split(","))
        server_local_idx = (
            int(os.getenv("CUDA_VISIBLE_DEVICES").split(",")[0]) // gpus_per_server
        )
        n_servers_per_node = max(1, self.n_gpus_per_node // gpus_per_server)
        ports_per_server = 40000 // n_servers_per_node
        port_range = (
            server_local_idx * ports_per_server + 10000,
            (server_local_idx + 1) * ports_per_server + 10000,
        )
        server_port, dist_init_port = find_free_ports(2, port_range)

        dist_init_addr = f"localhost:{dist_init_port}"
        host_ip = gethostip()

        cmd = SGLangConfig.build_cmd(
            self.config,
            tp_size=self.tp_size,
            base_gpu_id=0,
            host=host_ip,
            port=server_port,
            dist_init_addr=dist_init_addr,
        )
        self.server_process = launch_server_cmd(cmd)
        wait_for_server(f"http://{host_ip}:{server_port}")

        name = names.gen_servers(self.experiment_name, self.trial_name)
        name_resolve.add_subentry(name, f"{host_ip}:{server_port}")

        logger.info(f"SGLang server launched at: http://{host_ip}:{server_port}")
        return_code = self.server_process.wait()
        logger.info(
            f"SGLang server at http://{host_ip}:{server_port} exits, returncode={return_code}"
        )

    def __del__(self):
        if self.server_process and self.server_process.poll() is None:
            logger.info("Terminating SGLang server process...")
            self.server_process.terminate()
            self.server_process.wait()
            logger.info("SGLang server process terminated.")


def main(argv):
    config, _ = parse_cli_args(argv)
    config.sglang = to_structured_cfg(config.sglang, SGLangConfig)
    config.cluster = to_structured_cfg(config.cluster, ClusterSpecConfig)
    config.cluster.name_resolve = to_structured_cfg(
        config.cluster.name_resolve, NameResolveConfig
    )
    name_resolve.reconfigure(config.cluster.name_resolve)

    allocation_mode = config.allocation_mode
    allocation_mode = AllocationMode.from_str(allocation_mode)
    assert allocation_mode.type_ == AllocationType.DECOUPLED_SGLANG
    tp_size = allocation_mode.gen_tp_size

    sglang_server = SGLangServerWrapper(
        config.experiment_name,
        config.trial_name,
        config.sglang,
        tp_size,
        n_gpus_per_node=config.cluster.n_gpus_per_node,
    )
    sglang_server.run()


if __name__ == "__main__":
    main(sys.argv[1:])
