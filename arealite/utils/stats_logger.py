import getpass
import os
import time
from typing import Dict

import torch.distributed as dist
import wandb
from tensorboardX import SummaryWriter

from arealite.api.cli_args import StatsLoggerConfig
from arealite.api.io_struct import FinetuneSpec
from realhf.api.core.data_api import tabulate_stats
from realhf.base import logging


class StatsLogger:

    def __init__(self, config: StatsLoggerConfig, ft_spec: FinetuneSpec):
        self.logger = logging.getLogger("StatsLogger", "system")
        self.config = config
        self.ft_spec = ft_spec
        self.init()

    def init(self):
        if dist.is_initialized() and dist.get_rank() != 0:
            return

        self.start_time = time.perf_counter()
        # wandb init, connect to remote wandb host
        if self.config.wandb.mode != "disabled":
            wandb.login()
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
            config=self.config.wandb.config,
            dir=self.get_log_path(self.config),
            force=True,
            id=f"{self.config.experiment_name}_{self.config.trial_name}_train",
            resume="allow",
            settings=wandb.Settings(start_method="fork"),
        )
        # tensorboard logging
        self.summary_writer = None
        if self.config.tensorboard.path is not None:
            self.summary_writer = SummaryWriter(log_dir=self.config.tensorboard.path)

    def close(self):
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        self.info(
            f"Training completes! Total time elapsed {time.monotonic() - self.start_time:.2f}."
        )
        wandb.finish()
        if self.summary_writer is not None:
            self.summary_writer.close()

    def commit(self, epoch: int, step: int, global_step: int, data: Dict):
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        self.info(
            f"Epoch {epoch+1}/{self.ft_spec.total_train_epochs} "
            f"Step {step+1}/{self.ft_spec.steps_per_epoch} "
            f"Train step {global_step + 1}/{self.ft_spec.total_train_steps} done."
        )
        self.info("Stats:")
        self.print_stats(data)
        wandb.log(data, step=global_step)
        if self.summary_writer is not None:
            for key, val in data.items():
                self.summary_writer.add_scalar(f"{key}", val, global_step)

    def print_stats(self, stats: Dict[str, float]):
        self.info("\n" + tabulate_stats(stats))

    @staticmethod
    def get_log_path(config: StatsLoggerConfig):
        path = f"{config.fileroot}/logs/{getpass.getuser()}/{config.experiment_name}/{config.trial_name}"
        os.makedirs(path, exist_ok=True)
        return path

    def info(self, msg: str, *args, **kwargs):
        if dist.is_initialized() and dist.get_rank() > 0:
            return
        self.logger.info(msg, *args, **kwargs)

    def debug(self, msg: str, *args, **kwargs):
        if dist.is_initialized() and dist.get_rank() > 0:
            return
        self.logger.debug(msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs):
        if dist.is_initialized() and dist.get_rank() > 0:
            return
        self.logger.critical(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        if dist.is_initialized() and dist.get_rank() > 0:
            return
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        if dist.is_initialized() and dist.get_rank() > 0:
            return
        self.logger.error(msg, *args, **kwargs)
