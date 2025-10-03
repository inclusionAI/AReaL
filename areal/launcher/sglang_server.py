import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import psutil
import requests

from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import (
    ClusterSpecConfig,
    NameResolveConfig,
    SGLangConfig,
    parse_cli_args,
    to_structured_cfg,
)
from areal.platforms import current_platform
from areal.utils import logging, name_resolve, names
from areal.utils.launcher import TRITON_CACHE_PATH
from areal.utils.network import find_free_ports, gethostip

logger = logging.getLogger("SGLangServer Wrapper")


# Copied from SGLang
def kill_process_tree(parent_pid, include_parent: bool = True, skip_pid: int = None):
    """Kill the process and all its child processes."""
    # Remove sigchld handler to avoid spammy logs.
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)

    if parent_pid is None:
        parent_pid = os.getpid()
        include_parent = False

    try:
        itself = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        return

    children = itself.children(recursive=True)
    for child in children:
        if child.pid == skip_pid:
            continue
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass

    if include_parent:
        try:
            if parent_pid == os.getpid():
                itself.kill()
                sys.exit(0)

            itself.kill()

            # Sometime processes cannot be killed with SIGKILL (e.g, PID=1 launched by kubernetes),
            # so we send an additional signal to kill them.
            itself.send_signal(signal.SIGQUIT)
        except psutil.NoSuchProcess:
            pass


def launch_server_cmd(command: str) -> subprocess.Popen:
    """
    Execute a shell command and return its process handle.
    """
    # Replace newline continuations and split the command string.
    command = command.replace("\\\n", " ").replace("\\", " ")
    logger.info(f"Launch command: {command}")
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
        allocation_mode: AllocationMode,
        n_gpus_per_node: int,
    ):
        self.experiment_name = experiment_name
        self.trial_name = trial_name
        self.config = sglang_config
        self.allocation_mode = allocation_mode
        self.server_process = None
        self.n_gpus_per_node = n_gpus_per_node

    def run(self):
        gpus_per_server = self.allocation_mode.gen_instance_size
        cross_nodes = False
        if gpus_per_server > self.n_gpus_per_node:
            assert (
                gpus_per_server % self.n_gpus_per_node == 0
            ), "Cross-nodes SGLang only supports utilizing all gpus in one node"
            cross_nodes = True
            node_rank = int(os.environ["AREAL_SGLANG_MULTI_NODE_RANK"])
            master_addr = os.environ["AREAL_SGLANG_MULTI_NODE_MASTER_ADDR"]
            master_port = int(os.environ["AREAL_SGLANG_MULTI_NODE_MASTER_PORT"])
        else:
            node_rank = 0
            master_addr = None
            master_port = None

        n_servers_per_node = max(1, self.n_gpus_per_node // gpus_per_server)
        n_nodes_per_server = max(1, gpus_per_server // self.n_gpus_per_node)

        if current_platform.device_control_env_var in os.environ:
            visible = os.getenv(current_platform.device_control_env_var).split(",")
            n_visible_devices = len(visible)
            n_servers_per_proc = max(1, n_visible_devices // gpus_per_server)
            server_idx_offset = min(list(map(int, visible))) // gpus_per_server
        else:
            n_servers_per_proc = n_servers_per_node
            server_idx_offset = 0

        # Separate ports used by each server in the same node
        # ports range (10000, 50000)
        ports_per_server = 40000 // n_servers_per_node
        launch_server_args = []
        server_addresses = []
        base_random_seed = self.config.random_seed
        for server_local_idx in range(
            server_idx_offset, server_idx_offset + n_servers_per_proc
        ):
            port_range = (
                server_local_idx * ports_per_server + 10000,
                (server_local_idx + 1) * ports_per_server + 10000,
            )
            engine_ports = os.getenv("ENGINE_PORTS", "")
            server_port, dist_init_port = 0, 0
            if engine_ports != "":
                ports = engine_ports.split(",")
                if len(ports) == 2:
                    server_port = int(ports[0])
                    dist_init_port = int(ports[1])
                    print(f"SGLang server get ports from env, engine_ports: {engine_ports}")
            else:
                server_port, dist_init_port = find_free_ports(2, port_range)

            if cross_nodes:
                n_nodes = n_nodes_per_server
                dist_init_addr = f"{master_addr}:{master_port}"
            else:
                n_nodes = 1
                dist_init_addr = f"localhost:{dist_init_port}"

            host_ip = gethostip()

            base_gpu_id = (server_local_idx - server_idx_offset) * gpus_per_server
            self.config.random_seed = base_random_seed + server_local_idx
            cmd = SGLangConfig.build_cmd(
                self.config,
                tp_size=self.allocation_mode.gen.tp_size,
                base_gpu_id=base_gpu_id,
                host=host_ip,
                port=server_port,
                dist_init_addr=dist_init_addr,
                n_nodes=n_nodes,
                node_rank=node_rank,
            )
            launch_server_args.append((cmd, host_ip, server_port, node_rank))
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
                        f"SGLang server {server_addresses[i]} exits, returncode={return_code}"
                    )
                    all_alive = False
                    break

            if not all_alive:
                for i, process in enumerate(server_processes):
                    if process.poll() is None:
                        process.terminate()
                        process.wait()
                        logger.info(
                            f"SGLang server process{server_addresses[i]} terminated."
                        )

            time.sleep(1)

    def launch_one_server(self, cmd, host_ip, server_port, node_rank):
        server_process = launch_server_cmd(cmd)
        wait_for_server(f"http://{host_ip}:{server_port}")
        if node_rank == 0:
            name = names.gen_servers(self.experiment_name, self.trial_name)
            name_resolve.add_subentry(name, f"{host_ip}:{server_port}")
        logger.info(f"SGLang server launched at: http://{host_ip}:{server_port}")
        return server_process


def launch_sglang_server(argv):
    config, _ = parse_cli_args(argv)
    config.sglang = to_structured_cfg(config.sglang, SGLangConfig)
    config.cluster = to_structured_cfg(config.cluster, ClusterSpecConfig)
    config.cluster.name_resolve = to_structured_cfg(
        config.cluster.name_resolve, NameResolveConfig
    )
    name_resolve.reconfigure(config.cluster.name_resolve)

    allocation_mode = config.allocation_mode
    allocation_mode = AllocationMode.from_str(allocation_mode)
    assert allocation_mode.gen_backend == "sglang"

    sglang_server = SGLangServerWrapper(
        config.experiment_name,
        config.trial_name,
        config.sglang,
        allocation_mode,
        n_gpus_per_node=config.cluster.n_gpus_per_node,
    )
    sglang_server.run()


def main(argv):
    try:
        launch_sglang_server(argv)
    finally:
        kill_process_tree(os.getpid())


if __name__ == "__main__":
    main(sys.argv[1:])
