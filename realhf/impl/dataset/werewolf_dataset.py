import json
from typing import Callable, Dict, List, Optional

import torch
import torch.utils.data

from realhf.api.core import data_api
from realhf.base import logging

logger = logging.getLogger("Werewolf Dataset")

class WerewolfDataset(torch.utils.data.Dataset):
    """Simple dataset for the werewolf game.

    Each element of the dataset should contain:
        {
            "id": str,        # unique identifier
            "prompt": str     # initial prompt for the agent
        }
    """

    def __init__(
        self,
        util: data_api.DatasetUtility,
        max_length: Optional[int] = None,
        dataset_path: Optional[str] = None,
        dataset_builder: Optional[Callable[[], List[Dict]]] = None,
    ):
        self._util = util
        data = data_api.load_shuffle_split_dataset(util, dataset_path, dataset_builder)

        prompts = [x["prompt"] for x in data]
        self.ids = [str(x.get("id", i)) for i, x in enumerate(data)]
        util.tokenizer.padding_side = "left"
        enc = util.tokenizer(
            prompts,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_length=True,
            return_attention_mask=False,
        )
        self.prompt_lengths = enc["length"]
        self.prompts = enc["input_ids"]
        logger.info(f"Number of prompts in the dataset: {len(self.prompts)}")

    @property
    def util(self):
        return self._util

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return data_api.SequenceSample.from_default(
            ids=[self.ids[idx]],
            seqlens=[self.prompt_lengths[idx]],
            data=dict(packed_prompts=torch.tensor(self.prompts[idx], dtype=torch.long)),
        )


# Register dataset
if not __name__ == "__main__":
    data_api.register_dataset("werewolf_dataset", WerewolfDataset)