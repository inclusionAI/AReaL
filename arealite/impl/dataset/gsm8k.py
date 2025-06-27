from typing import Dict, List

import torch
from datasets import Dataset


def process_gsm8k_rl_dataset(dataset: Dataset, tokenizer, reward_mode):
    def process_example(example, idx):
        # Add query_id column
        example["query_id"] = str(idx)
        example["prompt"] = example["question"]

        # used by the reward function
        example["method"] = reward_mode
        return example

    dataset = dataset.map(
        lambda example, idx: process_example(example, idx),
        with_indices=True,
    )
    return dataset.map(
        lambda x: tokenizer(x["question"], return_attention_mask=False), batched=True
    )


def process_gsm8k_sft_dataset(dataset: Dataset, tokenizer):
    def process_example(example, idx):
        # Add query_id column
        example["query_id"] = str(idx)
        example["prompt"] = example["question"]
        example["seq"] = example["prompt"] + example["answer"] + tokenizer.eos_token
        return example

    dataset = dataset.map(
        lambda example, idx: process_example(example, idx),
        with_indices=True,
    )

    def _tokenize(example):
        example["prompt"] = tokenizer(example["prompt"], return_attention_mask=False)[
            "input_ids"
        ]
        example["seq"] = tokenizer(example["seq"], return_attention_mask=False)[
            "input_ids"
        ]
        return example

    dataset = dataset.map(lambda x: _tokenize(x), batched=True)
    return dataset
