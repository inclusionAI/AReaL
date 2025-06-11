import os

import hydra
from torch.distributed.elastic.multiprocessing.errors import record

from realhf.experiments.common.utils import AllocationMode
from refactoring.api.cli_args import TrainingArgs
from refactoring.api.llm_server_api import LLMServerFactory
from refactoring.api.trainer_api import TrainerFactory


@record
@hydra.main(version_base=None)
def main(args: TrainingArgs):
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    alloc = AllocationMode.from_str(args.allocation_mode)

    if rank < alloc.get_gen_size():
        server_factory = LLMServerFactory(args)
        server = server_factory.make_server(args.inf_service)
        server.start()
        return

    trainer_factory = TrainerFactory(args)
    trainer = trainer_factory.make_trainer(args.trainer)
    trainer.train()


if __name__ == "__main__":
    main()
