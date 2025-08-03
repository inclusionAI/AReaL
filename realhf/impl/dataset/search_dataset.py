# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import json
import os
import sys
import traceback
from collections import defaultdict
from typing import Callable, Dict, Hashable, List, Optional

import numpy as np
import torch.utils.data

from realhf.api.core import data_api
from realhf.base import logging

logger = logging.getLogger("Web Search Agent Dataset")

def load_metadata(dataset_path):
    data=[json.loads(ff) for ff in open(dataset_path)]
    for d in data:
        if "idx" in d:
            d["idx"] = str(d["idx"])
        else:
            d["idx"] = str(d["id"])
    id2info = {d["idx"]: d for d in data}
    return id2info

PROMPT_TEMPLATES = {
    "v0": "A conversation between User and Assistant. The user asks a question, and the Assistant answers it. The Assistant thinks about the reasoning process in the mind, calls a search engine to find necessary information, and provides the user with the answer. The Assistant conducts search by <search> query </search>, and it will return the top search results between <information> and </information>. The reasoning processes are enclosed within <think> </think>. Finally, the Assistant provides answer inside <answer> and </answer>, i.e. <answer> answer here </answer>. User: {prompt}. Assistant: \n<think>",
    "v1": "A conversation between User and Assistant. The user asks a question, and the Assistant answers it. The Assistant analyzes the given question and information in the mind, retains important relevant information, calls a search engine to find necessary information, accesses web pages with certain urls, and provides the user with the answer. The Assistant conducts search by <search> query </search>, access cerain url by <access> url </access>, and the top search results and url page will be returned between <information> and </information>.  The reasoning processes are enclosed within <think> </think>. Finally, the Assistant provides answer inside <answer> and </answer>, i.e. <answer> answer here </answer>. If there are multiple queries, ensure all answers are enclosed within <answer> </answer>, seperated with comma. Note that when the Assistant finds the question is invalid, e.g. no answer could match all information in the question, the Assistant replies with '<answer> the question is invalid. </answer>'. User: {prompt}. The language of your answer should align with the question. Assistant: \n<think>",
    "feimo-v0": '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nAnswer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {prompt}\n<|im_end|>\n<|im_start|>assistant\n',
    "reasoning-v0": "{prompt}",
}

class WebSearchDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        util: data_api.DatasetUtility,
        max_length: Optional[int] = None,
        dataset_path: Optional[str] = None,
        dataset_builder: Optional[Callable[[], List[Dict]]] = None,
        filter_threshold: float = 1e4,
        max_filter_percentage: float = 0.0,
        prompt_type: str = "v0",
        valid_inst_ratio: float = 0.0,
    ):
    
        self._util = util
        self.max_length = max_length
        self.valid_inst_ratio = valid_inst_ratio
        self.prompt_type = prompt_type
        self.tokenizer = util.tokenizer

        data = data_api.load_shuffle_split_dataset(util, dataset_path, dataset_builder)

        prompt_template = PROMPT_TEMPLATES[prompt_type]
        prompts_str = [prompt_template.format(prompt=x["question"]) for x in data]
        self.ids = [x.get("idx", x.get("id", None)) for x in data]
        util.tokenizer.padding_side = "left"
        prompt_encodings = util.tokenizer(
            prompts_str,
            truncation=True,
            # max_length=max_length,
            padding=False,
            return_length=True,
            return_attention_mask=False,
        )
        
        logger.info(f"{len(data)} samples, checking lengths (max_length={max_length})")
        indices = [
            i for i, x in enumerate(prompt_encodings["length"]) if x <= max_length
        ]
        logger.info(
            f"{len(indices)} samples remain"
        )
        self.prompt_lengths = [int(prompt_encodings["length"][idx]) for idx in indices]
        self.prompts = [prompt_encodings["input_ids"][idx] for idx in indices]
        self.ids = [
            str(self.ids[idx]) + f"@idx:{idx}-{util.dp_rank}" for idx in indices
        ]

        
        assert all(len(x) == l for x, l in zip(self.prompts, self.prompt_lengths))

        logger.info(f"Number of prompts in the dataset: {len(self.prompts)}")

        self.active_indices = list(range(len(self.prompts)))
        self.filter_threshold = filter_threshold
        self.max_filter_percentage = max_filter_percentage

    @property
    def util(self):
        return self._util

    def __len__(self):
        return len(self.active_indices)

    def __getitem__(self, idx):
        idx = self.active_indices[idx]
        token_ids = self.prompts[idx]
        prompt_length = self.prompt_lengths[idx]
        if self.valid_inst_ratio > 0 and self.prompt_type in ["v1"]:
            if np.random.uniform(0, 1) < self.valid_inst_ratio:
                prompt = self.tokenizer.decode(self.prompts[idx], skip_special_tokens=False)
                prompt = prompt.replace("Note that when the Assistant finds the question is invalid, e.g. no answer could match all information in the question, the Assistant replies with '<answer> the question is invalid. </answer>'. ", "\n\nNote: the question is a valid question and you should try to find a correct answer. ")
                token_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
                prompt_length = len(token_ids)

        data = dict(
            packed_prompts=torch.tensor(token_ids, dtype=torch.long),
        )
        return data_api.SequenceSample.from_default(
            ids=[self.ids[idx]],
            seqlens=[prompt_length],
            data=data,
        )


    def filter(self, eval_scores: Dict[Hashable, float]):
        # Get all data indices that have a higher score than the threshold.
        idx2scores_to_remove = {}
        for pop_idx, idx in enumerate(self.active_indices):
            data_id = self.ids[idx]
            if data_id not in eval_scores:
                continue
            if eval_scores[data_id] > self.filter_threshold:
                idx2scores_to_remove[pop_idx] = eval_scores[data_id]

        # Control the number of samples to be removed according to max_filter_percentage.
        n = int(len(self.active_indices) * self.max_filter_percentage)
        indices_to_remove = sorted(
            idx2scores_to_remove.keys(),
            key=lambda x: idx2scores_to_remove[x],
            reverse=True,
        )[:n]

        for pop_idx in sorted(indices_to_remove, reverse=True):
            self.active_indices.pop(pop_idx)
        logger.info(
            f"Math prompt dataset DP rank {self.util.dp_rank} filtered"
            f" {len(indices_to_remove)} samples, {len(self.active_indices)} samples remain. "
            f"Original dataset size: {len(self.prompts)}. "
            f"Filter threshold: {self.filter_threshold}. "
            f"Max filter percentage: {self.max_filter_percentage}. "
            f"Current number of eval scores: {len(eval_scores)}."
        )


if not __name__ == "__main__":
    data_api.register_dataset("web-search", WebSearchDataset)