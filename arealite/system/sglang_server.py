# Copyright 2025 Ant Group Inc.
# Licensed under the Apache License, Version 2.0

import os
import subprocess
import sys
import traceback
from pathlib import Path

import requests

from arealite.api.cli_args import LLMServiceConfig, SGLangConfig
from arealite.api.io_struct import AllocationMode, LLMServerInfo
from arealite.api.llm_server_api import LLMServer
from realhf.base import gpu_utils, logging, network, pkg_version

logger = logging.getLogger(__name__)


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

    def __init__(self, args, service_config: LLMServiceConfig):
        super().__init__(args, service_config)
        self.server_info: LLMServerInfo | None = None
        self.base_gpu_id = 0
        self.config = args.rollout.sglang

        self.alloc_mode = AllocationMode.from_str(args.allocation_mode)

    def _resolve_base_gpu_id(self):
        # Determine GPU configuration
        import ray

        tp_size = self.alloc_mode.gen_tp_size
        pp_size = self.alloc_mode.gen_pp_size
        mp_size = tp_size * pp_size
        if ray.is_initialized():
            self.base_gpu_id = 0
        elif "CUDA_VISIBLE_DEVICES" in os.environ:
            if len(os.environ["CUDA_VISIBLE_DEVICES"]) == 1:
                self.base_gpu_id = int(os.environ["CUDA_VISIBLE_DEVICES"])
            elif len(os.environ["CUDA_VISIBLE_DEVICES"]) == mp_size:
                self.base_gpu_id = int(os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0])
            else:
                logger.warning(
                    f"Unknown how to resolve cuda visible devices: {os.environ['CUDA_VISIBLE_DEVICES']}, "
                    f"setting base_gpu_id to 0."
                )
                self.base_gpu_id = 0
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
                map(str, range(gpu_utils.gpu_count()))
            )
        elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            # torchrun
            self.base_gpu_id = int(os.environ["RANK"]) % gpu_utils.gpu_count()
        elif gpu_utils.gpu_count() == mp_size:
            self.base_gpu_id = 0
        else:
            logger.warning("Unknown GPU configuration, setting base_gpu_id to 0. ")
            self.base_gpu_id = 0

    def launch_server(self) -> LLMServerInfo | None:
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
        tp_size = self.alloc_mode.gen_tp_size
        cmd = SGLangConfig.build_cmd(
            sglang_config=self.config,
            model_path=self.args.rollout.model_path,
            tp_size=tp_size,
            base_gpu_id=self.base_gpu_id,
            dist_init_addr=f"{host}:{nccl_port}",
            served_model_name=self.service_config.served_model_name,
            skip_tokenizer_init=False,
        )
        # Launch process
        full_command = f"{cmd} --port {server_port}"
        full_command = full_command.replace("\\\n", " ").replace("\\", " ")
        self.process = subprocess.Popen(
            full_command.split(),
            text=True,
            stdout=sys.stdout,
            stderr=subprocess.STDOUT,
        )
        # Create server info
        self.server_info = LLMServerInfo(
            server_id=self.server_id,
            host=host,
            port=server_port,
            status="starting",
            version=0,
        )
        return self.server_info

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
                f"{base_url}/metrics",
                timeout=30,
            )
            if response.status_code != 200:
                return False
            # Update server load
            for line in response.text.split("\n"):
                if line.startswith("sglang:num_running_reqs"):
                    self.load = float(line.split(" ")[1])
                    break
            return True
        except requests.exceptions.RequestException:
            return False
