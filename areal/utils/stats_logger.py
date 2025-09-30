import getpass
import os
import time
from typing import Dict, List

import swanlab
import torch.distributed as dist
import wandb
from megatron.core import parallel_state as mpu
from tensorboardX import SummaryWriter

from areal.api.cli_args import BaseExperimentConfig, StatsLoggerConfig
from areal.api.io_struct import FinetuneSpec
from areal.utils import logging
from areal.utils.printing import tabulate_stats

logger = logging.getLogger("StatsLogger", "system")


class StatsLogger:

    def __init__(self, config: BaseExperimentConfig, ft_spec: FinetuneSpec):
        if isinstance(config, StatsLoggerConfig):
            raise ValueError(
                "Passing config.stats_logger as the config is deprecated. "
                "Please pass the full config instead."
            )
        self.exp_config = config
        self.config = config.stats_logger
        self.ft_spec = ft_spec
        self.init()

        self._last_commit_step = 0

    def init(self):
        if dist.is_initialized() and dist.get_rank() != 0:
            return

        self.start_time = time.perf_counter()
        # wandb init, connect to remote wandb host
        if self.config.wandb.mode != "disabled":
            wandb.login()

        if self.config.wandb.wandb_base_url:
            os.environ["WANDB_API_KEY"] = self.config.wandb.wandb_api_key
        if self.config.wandb.wandb_api_key:
            os.environ["WANDB_BASE_URL"] = self.config.wandb.wandb_base_url

        suffix = self.config.wandb.id_suffix
        if suffix == "timestamp":
            suffix = time.strftime("%Y_%m_%d_%H_%M_%S")

        wandb.init(
            mode=self.config.wandb.mode,
            entity=self.config.wandb.entity,
            project=self.config.wandb.project or self.config.experiment_name,
            name=self.config.wandb.name or self.config.trial_name,
            job_type=self.config.wandb.job_type,
            group=self.config.wandb.group
            or f"{self.config.experiment_name}_{self.config.trial_name}",
            notes=self.config.wandb.notes,
            tags=self.config.wandb.tags,
            config=self.exp_config,  # save all experiment config to wandb
            dir=self.get_log_path(self.config),
            force=True,
            id=f"{self.config.experiment_name}_{self.config.trial_name}_{suffix}",
            resume="allow",
            settings=wandb.Settings(start_method="fork"),
        )

        swanlab_config = self.config.swanlab
        if swanlab_config.mode != "disabled":
            if swanlab_config.api_key:
                swanlab.login(swanlab_config.api_key)
            else:
                swanlab.login()

        swanlab_config = self.config.swanlab
        swanlab.init(
            project=swanlab_config.project or self.config.experiment_name,
            experiment_name=swanlab_config.name or self.config.trial_name + "_train",
            # NOTE: change from swanlab_config.config to log all experiment config, to be tested
            config=self.exp_config,
            logdir=self.get_log_path(self.config),
            mode=swanlab_config.mode,
        )
        # tensorboard logging
        self.summary_writer = None
        if self.config.tensorboard.path is not None:
            self.summary_writer = SummaryWriter(log_dir=self.config.tensorboard.path)

    def state_dict(self):
        return {
            "last_commit_step": self._last_commit_step,
        }

    def load_state_dict(self, state_dict):
        self._last_commit_step = state_dict["last_commit_step"]

    def close(self):
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        logger.info(
            f"Training completes! Total time elapsed {time.monotonic() - self.start_time:.2f}."
        )
        wandb.finish()
        swanlab.finish()
        if self.summary_writer is not None:
            self.summary_writer.close()

    def commit(self, epoch: int, step: int, global_step: int, data: Dict | List[Dict]):
        if dist.is_initialized() and mpu.is_initialized():
            if mpu.get_pipeline_model_parallel_world_size() > 1:
                # log info only exist in last pipeline rank
                data_list = [data]
                dist.broadcast_object_list(
                    data_list,
                    src=mpu.get_pipeline_model_parallel_last_rank(),
                    group=mpu.get_pipeline_model_parallel_group(),
                )
                data = data_list[0]
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        logger.info(
            f"Epoch {epoch+1}/{self.ft_spec.total_train_epochs} "
            f"Step {step+1}/{self.ft_spec.steps_per_epoch} "
            f"Train step {global_step + 1}/{self.ft_spec.total_train_steps} done."
        )
        if isinstance(data, Dict):
            data = [data]
        log_step = max(global_step, self._last_commit_step + 1)
        for i, item in enumerate(data):
            logger.info(f"Stats ({i+1}/{len(data)}):")
            self.print_stats(item)
            wandb.log(item, step=log_step + i)
            swanlab.log(item, step=log_step + i)
            if self.summary_writer is not None:
                for key, val in item.items():
                    self.summary_writer.add_scalar(f"{key}", val, log_step + i)
        self._last_commit_step = log_step + len(data) - 1

    def print_stats(self, stats: Dict[str, float]):
        logger.info("\n" + tabulate_stats(stats))

    @staticmethod
    def get_log_path(config: StatsLoggerConfig):
        path = f"{config.fileroot}/logs/{getpass.getuser()}/{config.experiment_name}/{config.trial_name}"
        os.makedirs(path, exist_ok=True)
        return path
