import os
import subprocess
import sys
from pathlib import Path

import requests

from arealite.api.cli_args import LLMServiceConfig, TrainingArgs
from arealite.api.io_struct import LLMServerInfo
from arealite.api.llm_server_api import LLMServer
from realhf.api.cli_args import SGLangConfig
from realhf.base import gpu_utils, logging, name_resolve, names, network, pkg_version
from realhf.base.cluster import spec as cluster_spec

logger = logging.getLogger(__name__)


def execute_shell_command(command: str) -> subprocess.Popen:
    """Execute a shell command and return its process handle."""
    command = command.replace("\\\n", " ").replace("\\", " ")
    parts = command.split()
    return subprocess.Popen(
        parts,
        text=True,
        stdout=sys.stdout,
        stderr=subprocess.STDOUT,
    )


def apply_sglang_path():
    """Apply SGLang patch if available."""
    p = Path(os.path.dirname(__file__))
    patch_path = str(
        p.parent.parent.parent
        / "patch"
        / "sglang"
        / f"v{pkg_version.get_version('sglang')}.patch"
    )

    target_path = ""
    try:
        sglang_meta = subprocess.check_output(
            "python3 -m pip show sglang", shell=True
        ).decode("ascii")
        for line in sglang_meta.split("\n"):
            line = line.strip()
            if line.startswith("Editable project location: "):
                target_path = str(Path(line.split(": ")[1]).parent)

        if target_path and Path(patch_path).exists():
            proc = subprocess.Popen(
                ["git", "apply", patch_path],
                cwd=target_path,
                stderr=sys.stdout,
                stdout=sys.stdout,
            )
            proc.wait()
            logger.info(f"Applied SGLang patch at {target_path}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass


class SGLangServer(LLMServer):
    """SGLang implementation of LLMServer."""

    def __init__(self, args: TrainingArgs, service_config: LLMServiceConfig):
        super().__init__(args, service_config)
        self.server_info: LLMServerInfo | None = None
        self.base_gpu_id = 0
        self.config = service_config.sglang

    def _resolve_base_gpu_id(self):
        # Determine GPU configuration
        import ray

        tp_size = self.service_config.parallel.tensor_parallel_size
        if ray.is_initialized():
            self.base_gpu_id = 0
        elif "CUDA_VISIBLE_DEVICES" in os.environ:
            if len(os.environ["CUDA_VISIBLE_DEVICES"]) == 1:
                self.base_gpu_id = int(os.environ["CUDA_VISIBLE_DEVICES"])
            elif len(os.environ["CUDA_VISIBLE_DEVICES"]) == tp_size:
                self.base_gpu_id = int(os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0])
            else:
                raise RuntimeError(
                    f"Unknown how to resolve cuda visible devices: {os.environ['CUDA_VISIBLE_DEVICE']}"
                )
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
                map(str, range(gpu_utils.gpu_count()))
            )
        elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            self.base_gpu_id = int(os.environ["RANK"]) % gpu_utils.gpu_count()
        else:
            raise RuntimeError("Cannot resolve base GPU ID.")

    def launch_server(self) -> LLMServerInfo | None:
        """Launch the SGLang server subprocess."""
        try:

            # Apply SGLang patch
            apply_sglang_path()

            self._resolve_base_gpu_id()

            # Get host and ports
            host_ip = network.gethostip()
            host = "localhost" if not self.config.enable_metrics else host_ip

            ports = network.find_multiple_free_ports(
                2,
                low=10000,
                high=60000,
                experiment_name=self.registry.expr_name,
                trial_name=self.registry.trial_name,
            )
            server_port = ports[0]
            nccl_port = ports[1]

            # Build command
            tp_size = self.service_config.parallel.tensor_parallel_size
            cmd = SGLangConfig.build_cmd(
                self.config,
                self.service_config.model_path,
                tp_size=tp_size,
                base_gpu_id=self.base_gpu_id,
                dist_init_addr=f"{host}:{nccl_port}",
            )

            # Launch process
            full_command = f"{cmd} --port {server_port}"
            self.process = execute_shell_command(full_command)

            # Create server info
            self.server_info = LLMServerInfo(
                server_id=self.server_id,
                host=host,
                port=server_port,
                status="starting",
            )

            logger.info(f"SGLang server launched at: http://{host}:{server_port}")
            return self.server_info

        except Exception as e:
            logger.error(f"Failed to launch SGLang server: {e}")
            return None

    def check_health(self) -> bool:
        """Check if the SGLang server is healthy."""
        if not self.server_info or not self.process:
            return False

        # Check if process is still running
        if self.process.poll() is not None:
            return False

        try:
            # Check server endpoint
            base_url = f"http://{self.server_info.host}:{self.server_info.port}"
            response = requests.get(
                f"{base_url}/v1/models",
                headers={"Authorization": "Bearer None"},
                timeout=5,
            )
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
