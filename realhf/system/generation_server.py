import os
import time

from realhf.api.cli_args import SGLangConfig
from realhf.api.core.system_api import GenerationServer as GenerationServerConfig
from realhf.base import gpu_utils, logging, name_resolve, names, network, seeding
from realhf.system.worker_base import PollResult, Worker

logger = logging.getLogger(__name__)


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
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            self.base_gpu_id = int(os.environ["CUDA_VISIBLE_DEVICES"])
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
                map(str, range(gpu_utils.gpu_count()))
            )

        self.server_process = None
        self.server_addr = None

        return config.worker_info

    def launch_server_subprocess(self):
        config = self.config

        assert config.backend_type == "sglang"
        cmd = SGLangConfig.build_cmd(
            config.backend_args,
            config.model_path,
            tp_size=config.tp_size,
            server_index=self.worker_index,
            base_gpu_id=self.base_gpu_id,
        )
        from sglang.utils import launch_server_cmd, wait_for_server

        host_ip = network.gethostip()
        host = "localhost" if not config.backend_args.enable_metrics else host_ip

        # TODO: handle launching error and retry
        self.server_process, self.server_port = launch_server_cmd(cmd)
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

        # TODO: we may want to collect some metrics from the server
        time.sleep(0.05)

        return PollResult(0, 0)

    def _exit_hook(self, exit_status):
        if self.server_process is not None and self.config.backend_type == "sglang":
            from sglang.utils import terminate_process

            terminate_process(self.server_process)
