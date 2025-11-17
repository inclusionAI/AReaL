import os
import signal
import traceback
from concurrent.futures import Future, as_completed

import torch
from datasets import Dataset
from torch.utils.data import Sampler

from areal.utils import logging

logger = logging.getLogger("Utils")

RL_TASKS = ["math", "code", "rlhf", "stem", "general", "logic", "ifeval", "swe"]

def wait_future_ordered(futures: list[Future], exit_on_exception: bool = False) -> list:
    results = [None] * len(futures)
    future_index_map = {future: i for i, future in enumerate(futures)}
    for future in as_completed(futures):
        index = future_index_map[future]
        try:
            results[index] = future.result()
        except Exception as e:
            logger.warning(f"Exception caught when waiting for future: {e}")
            logger.warning(traceback.format_exc())
            if exit_on_exception:
                logger.info("Exiting due to exception in future.")
                os.kill(os.getpid(), signal.SIGTERM)
            else:
                raise e
    return results


def process_rl_dataset(dataset: Dataset):
    def process(sample):
        messages = [{"role": "user", "content": sample["prompt"]}]
        return {"messages": messages}

    dataset = dataset.map(process).remove_columns(["prompt"])
    return dataset


def get_shuffle_indices(size: int, seed: int):
    """Generate shuffled indices given seed and (dataset) size."""
    g = torch.Generator()
    g.manual_seed(seed)
    shuffle_idx = torch.randperm(size, generator=g).tolist()  # type: ignore[arg-type]
    return shuffle_idx


class ShuffleSampler(Sampler):
    def __init__(self, data_source, seed=42):
        self.data_source = data_source
        self.shuffle_indices = get_shuffle_indices(size=len(data_source), seed=seed)

    def __iter__(self):
        return iter(self.shuffle_indices)

    def __len__(self):
        return len(self.data_source)
