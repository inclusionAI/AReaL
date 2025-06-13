import os

import hydra
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record

from arealite.api.cli_args import TrainingArgs
from arealite.api.llm_server_api import LLMServerFactory
from arealite.api.trainer_api import TrainerFactory
from realhf.experiments.common.utils import AllocationMode


@record
@hydra.main(version_base=None)
def main(args: TrainingArgs):
    dist.init_process_group("nccl|gloo")
    trainer_factory = TrainerFactory(args)
    trainer = trainer_factory.make_trainer(args.trainer)
    trainer.train()


if __name__ == "__main__":
    main()
