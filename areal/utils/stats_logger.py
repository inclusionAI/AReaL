import getpass
import os
import time
from typing import Dict, List

import torch.distributed as dist
import wandb
from tensorboardX import SummaryWriter

from areal.api.cli_args import StatsLoggerConfig
from areal.api.io_struct import FinetuneSpec
from realhf.api.core.data_api import tabulate_stats
from realhf.base import logging

logger = logging.getLogger("StatsLogger", "system")


class StatsLogger:

    def __init__(self, config: StatsLoggerConfig, ft_spec: FinetuneSpec):
        self.config = config
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
        logger.info(
            f"Training completes! Total time elapsed {time.monotonic() - self.start_time:.2f}."
        )
        wandb.finish()
        if self.summary_writer is not None:
            self.summary_writer.close()

    def commit(self, epoch: int, step: int, global_step: int, data: Dict | List[Dict]):
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
