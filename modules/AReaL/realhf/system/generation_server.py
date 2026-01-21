import os
import subprocess
import sys
import time
from pathlib import Path

import ray
import requests

from realhf.api.cli_args import SGLangConfig
from realhf.api.core.system_api import ExpStatus
from realhf.api.core.system_api import GenerationServer as GenerationServerConfig
from realhf.base import (
    constants,
    gpu_utils,
    logging,
    name_resolve,
    names,
    network,
    pkg_version,
    seeding,
)
from realhf.system.worker_base import PollResult, Worker

logger = logging.getLogger(__name__)


def execute_shell_command(command: str) -> subprocess.Popen:
    """
    Execute a shell command and return its process handle.
    """
    # Replace newline continuations and split the command string.
    command = command.replace("\\\n", " ").replace("\\", " ")
    parts = command.split()
    return subprocess.Popen(
        parts,
        text=True,
        stdout=sys.stdout,
        stderr=subprocess.STDOUT,
    )


def launch_server_cmd(command: str, port: int = 30000):
    """
    Launch the server using the given command.
    If no port is specified, a free port is reserved.
    """
    assert port is not None
    full_command = f"{command} --port {port}"
    process = execute_shell_command(full_command)
    return process, port


def terminate_process(process, port=None):
    """
    Terminate the process and, if a port was reserved, release it.
    """
    from sglang.srt.utils import kill_process_tree

    kill_process_tree(process.pid)


def wait_for_server(base_url: str, timeout: int = None) -> None:
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


PORT_CLEARANCE_PERIOD = 90


class GenerationServer(Worker):
    def _configure(self, config: GenerationServerConfig):
        self.config = config
        self.worker_index = config.worker_info.worker_index
        self.worker_count = config.worker_info.worker_count
        self.experiment_name = config.worker_info.experiment_name
        self.trial_name = config.worker_info.trial_name
        seeding.set_random_seed(
            config.base_seed, f"generation_server{self.worker_index}"
        )

        # Cancel the effect of CUDA device isolation
        if ray.is_initialized():
            self.base_gpu_id = 0
        elif "CUDA_VISIBLE_DEVICES" in os.environ:
            self.base_gpu_id = int(os.environ["CUDA_VISIBLE_DEVICES"])
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
                map(str, range(gpu_utils.gpu_count()))
            )
        else:
            servers_per_node = self.args.cluster.n_gpus_per_node // self.config.tp_size
            idx_on_this_node = self.worker_index % servers_per_node
            self.base_gpu_id = idx_on_this_node * self.config.tp_size

        self.server_process = None
        self.server_addr = None

        return config.worker_info

    def launch_server_subprocess(self):
        config = self.config

        assert config.backend_type == "sglang"

        host_ip = network.gethostip()
        host = "localhost" if not config.backend_args.enable_metrics else host_ip

        # NOTE: Ports returned by `find_multiple_free_ports` are unique,
        # but SGLang servers still encounter conflicts.
        # Use a clearance period to hack over this issue.
        servers_per_node = self.args.cluster.n_gpus_per_node // self.config.tp_size
        idx_on_this_node = self.worker_index % servers_per_node
        time.sleep(idx_on_this_node * PORT_CLEARANCE_PERIOD / servers_per_node)

        ports = network.find_multiple_free_ports(
            2,
            low=10000,
            high=60000,
            experiment_name=self.experiment_name,
            trial_name=self.trial_name,
            lockfile_root=os.path.join(constants.get_cache_path(self.args), "ports"),
        )
        server_port = ports[0]
        nccl_port = ports[1]

        cmd = SGLangConfig.build_cmd(
            config.backend_args,
            config.model_path,
            tp_size=config.tp_size,
            server_index=self.worker_index,
            base_gpu_id=self.base_gpu_id,
            dist_init_addr=f"{host}:{nccl_port}",
        )

        self.server_process, self.server_port = launch_server_cmd(cmd, port=server_port)
        self.server_addr = f"http://{host}:{self.server_port}"

        wait_for_server(self.server_addr)

        name = names.gen_servers(self.experiment_name, self.trial_name)
        name_resolve.add_subentry(name, self.server_addr)

        key = names.metric_server(
            self.experiment_name,
            self.trial_name,
            "sglang",
            f"server{self.worker_index}",
        )
        name_resolve.add(
            key, f"{host}:{self.server_port}", keepalive_ttl=None, delete_on_exit=True
        )

        logger.info(f"SGLang server launched at: {self.server_addr}")

    def _poll(self):
        if self.server_process is None:
            self.launch_server_subprocess()

        # Check experiment finish.
        name = names.experiment_status(
            constants.experiment_name(), constants.trial_name()
        )
        try:
            exp_status = name_resolve.wait(name, timeout=300)
            if exp_status != str(ExpStatus.RUNNING):
                self.exit()
                return PollResult(0, 0)
        except TimeoutError:
            raise TimeoutError(
                f"Waiting for experiment status timeout. "
                "This indicates that the master worker is not running. Exit the worker."
            )

        time.sleep(5)

        return PollResult(0, 0)

    def _exit_hook(self, exit_status):
        if self.server_process is not None and self.config.backend_type == "sglang":

            terminate_process(self.server_process)
