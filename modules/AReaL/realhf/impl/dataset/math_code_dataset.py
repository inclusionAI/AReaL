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
from realhf.utils import load_hf_or_local_file

logger = logging.getLogger("Math Code Dataset")


def check_math_metadata_entries(data):
    assert data["task"] == "math" or data["task"] == "stem"
    assert "query_id" in data
    data["query_id"] = str(data["query_id"])
    assert isinstance(data["prompt"], str)
    assert isinstance(data["solutions"], list)
    for sol in data["solutions"]:
        assert isinstance(sol, str)
    return data


def check_code_metadata_entries(data):
    assert data["task"] == "code"
    assert "query_id" in data
    data["query_id"] = str(data["query_id"])
    if "problem_id" not in data:
        data["problem_id"] = data["query_id"]
    assert isinstance(data["prompt"], str)
    case_size = sys.getsizeof(data["input_output"])
    if os.getenv("FUNCTIONCALL_SERVICE_DOMAIN", ""):
        assert (
            case_size < 500 * 1024
        ), f"'input_output' exceeds 500KB ({case_size} bytes). Use remote testcase instead."
    input_output = json.loads(data["input_output"])
    assert len(input_output["inputs"]) == len(input_output["outputs"])
    for inp, out in zip(input_output["inputs"], input_output["outputs"]):
        assert isinstance(inp, str) and isinstance(out, str), (
            inp,
            out,
            input_output.get("fn_name"),
        )
    return data


def load_metadata(path):
    assert str(path).endswith(".jsonl"), path
    path = load_hf_or_local_file(path)
    with open(path, "r") as f:
        data = [json.loads(l) for l in f.readlines()]

    id2info = {}
    omit_cnt = defaultdict(int)
    task_cnt = defaultdict(int)
    for d in data:
        assert d["query_id"] not in d, (d["task"], d["query_id"])
        try:
            if "task" not in d:
                d["task"] = "math"
                logger.warning(
                    f'Key "task" not found in the dataset. Use math as default task type.'
                )
            if d["task"] == "math" or d["task"] == "stem":
                d = check_math_metadata_entries(d)
            elif d["task"] == "code":
                d = check_code_metadata_entries(d)
        except Exception as e:
            logger.warning(
                f"Data validation failed: query_id {d['query_id']}. "
                f"Error: {traceback.format_exc()}. Omit it in the dataset."
            )
            omit_cnt[d["task"]] += 1
            continue
        id2info[d["query_id"]] = d
        task_cnt[d["task"]] += 1
    logger.warning(f"Number of ignored data: {dict(**omit_cnt)}")
    return id2info, task_cnt


class MATHCodePromptDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        util: data_api.DatasetUtility,
        max_length: Optional[int] = None,
        dataset_path: Optional[str] = None,
        dataset_builder: Optional[Callable[[], List[Dict]]] = None,
        filter_threshold: float = 1e4,
        max_filter_percentage: float = 0.0,
    ):
        """Required keys: prompt, query_id, task=math/code, solutions.

        For code dataset, they additionally require an "input_output" key.
        """
        self._util = util
        self.max_length = max_length

        id2info, task_cnt = load_metadata(dataset_path)

        data = data_api.load_shuffle_split_dataset(util, dataset_path, dataset_builder)

        prompts_str = [x["prompt"] for x in data]
        self.ids = [x["query_id"] for x in data]
        self.tasks_ids = [data_api.RL_TASKS.index(x["task"]) for x in data]
        if "scores" in data[0]:
            self.base_scores = [np.mean(x["scores"]) for x in data]
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
            f"{len(indices)} samples remain, among them {task_cnt['math']} are math data and {task_cnt['code']} are code data"
        )

        self.prompt_lengths = [int(prompt_encodings["length"][idx]) for idx in indices]
        self.prompts = [prompt_encodings["input_ids"][idx] for idx in indices]
        self.ids = [
            str(self.ids[idx]) + f"@idx:{idx}-{util.dp_rank}" for idx in indices
        ]
        self.tasks_ids = [self.tasks_ids[idx] for idx in indices]
        if "scores" in data[0]:
            self.base_scores = [self.base_scores[idx] for idx in indices]

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
        # print(self.base_scores)
        idx = self.active_indices[idx]
        data = dict(
            task_ids=torch.tensor([self.tasks_ids[idx]], dtype=torch.long),
            packed_prompts=torch.tensor(self.prompts[idx], dtype=torch.long),
        )
        if hasattr(self, "base_scores"):
            data["base_scores"] = torch.tensor(
                [self.base_scores[idx]], dtype=torch.float32
            )
        return data_api.SequenceSample.from_default(
            ids=[self.ids[idx]],
            seqlens=[self.prompt_lengths[idx]],
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
    data_api.register_dataset("math_code_prompt", MATHCodePromptDataset)
else:
    from transformers import AutoTokenizer

    dataset = MATHCodePromptDataset(
        data_api.DatasetUtility(
            seed=0,
            dp_rank=0,
            world_size=1,
            tokenizer=AutoTokenizer.from_pretrained(
                "/storage/openpsi/models/Qwen__Qwen2-1.5B-Instruct/"
            ),
        ),
        max_length=512,
        dataset_path="/storage/datasets/full_prompts_for_r1_distilled.jsonl",
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=data_api.SequenceSample.gather,
        # NOTE: This is *NOT* the actual batch size for training.
        # It is just a proper size to load data to workers.
        batch_size=4,
        shuffle=True,
    )
    print(f"size: {len(dataset)}")
    for d in dataloader:
        # print(d.ids)
        pass
