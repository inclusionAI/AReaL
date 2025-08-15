import itertools
import os
import sys

import torch
import torch.distributed as dist
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.api.io_struct import AllocationMode, FinetuneSpec, StepInfo, WeightUpdateMeta
from areal.dataset import get_custom_dataset
from areal.engine.ppo.actor import FSDPPPOActor
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.utils.device import log_gpu_stats
from areal.utils.evaluator import Evaluator
from areal.utils.recover import RecoverHandler
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger
from areal.workflow.werewolf import WerewolfWorkflow
from realhf.api.core.data_api import load_hf_tokenizer
from realhf.base import seeding, stats_tracker

from realhf.base import logging
logger = logging.getLogger("Inference Only Experiment")


def main(args):
    config, _ = load_expr_config(args, GRPOConfig)
    config: GRPOConfig

    rank = int(os.getenv("RANK"))
    world_size = int(os.getenv("WORLD_SIZE"))
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    seeding.set_random_seed(config.seed, key=f"trainer{rank}")

    train_dataset = get_custom_dataset(
        path=config.train_dataset.path,
        rank=rank,
        world_size=world_size,
        split="train",
        type=config.train_dataset.type,
        tokenizer=tokenizer,
    )

    train_dataloader = StatefulDataLoader(
        train_dataset,
        batch_size=config.train_dataset.batch_size // world_size,
        shuffle=config.train_dataset.shuffle,
        num_workers=config.train_dataset.num_workers,
        collate_fn=lambda x: x,
        drop_last=config.train_dataset.drop_last,
    )
    ft_spec = FinetuneSpec(
        total_train_epochs=config.total_train_epochs,
        dataset_size=len(train_dataloader) * config.train_dataset.batch_size,
        train_batch_size=config.train_dataset.batch_size,
    )

    rollout = RemoteSGLangEngine(config.rollout)
    rollout.initialize(None, ft_spec)

    # Since this experiment is for inference only, we do not need training loop
    logger.warning(f"Training placeholder loop start. No outputs are expected at all.")
    while True:
        # This is an intended deadlock!
        pass
    
    rollout.destroy()


if __name__ == "__main__":
    main(sys.argv[1:])


"""
vim ../AReaL-Lite/examples/lite/configs/gsm8k_grpo.yaml

sudo srun --mpi=pmi2 --ntasks=1 --gres=gpu:8 \
    --cpus-per-task=10 --mem=1500G --pty \
    singularity shell --nv --no-home --writable-tmpfs \
    --bind /storage:/storage /storage/openpsi/images/sglang-v0.4.9.post2-cu126-v2.sif

cd /storage/openpsi/users/xmy/inclusionAI/AReaL
pip install -e . --no-deps
export HF_ENDPOINT="https://hf-mirror.com"
export WANDB_API_KEY=local-667d8d7f101dad4eb9597d718d0c68f40e3792f9
export WANDB_BASE_URL=http://8.150.1.98:8080

python -m areal.launcher.local examples/lite/infer_only.py \
    --config examples/lite/configs/infer_only.yaml \
    stats_logger.wandb.mode=disabled
"""
