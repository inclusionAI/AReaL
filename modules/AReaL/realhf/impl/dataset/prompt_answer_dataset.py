# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import json
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.utils.data

from realhf.api.core import data_api
from realhf.base import logging

logger = logging.getLogger("Prompt Answer Dataset")


class PromptAnswerDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        util: data_api.DatasetUtility,
        max_length: int,
        dataset_path: Optional[str] = None,
        dataset_builder: Optional[Callable[[], List[Dict]]] = None,
        fill_to_max_length: bool = False,
    ):
        """A dataset with prompts and corresponding answers. Usually used for
        SFT.

        Args:
            util (api.data.DatasetUtility): .
            max_length (Optional[int], optional): The maximum length of each sequence in the batch.
            dataset_path (Optional[str], optional): Path to the dataset json/jsonl file.
                The json/jsonl file should be a list of dictionary. Each element in the list should have
                a key "prompt" and a key "answer". Defaults to None.
            dataset_builder (Optional[Callable[[], List[Dict]]], optional): Alternative to dataset_path.
                A callable that returns a list of dictionary. Defaults to None.
            fill_to_max_length (bool): Whether to fill sequences to the maximum length. If True,
                prompts will be left-filled with non-pad tokens. Only used for testing.
        """
        self._util = util
        tokenizer = self.util.tokenizer

        data = data_api.load_shuffle_split_dataset(util, dataset_path, dataset_builder)

        seqs = [x["prompt"] + x["answer"] + tokenizer.eos_token for x in data]
        self.ids = [x["id"] for x in data]
        prompts = [x["prompt"] for x in data]

        self.tokens = tokenizer(
            seqs,
            truncation=True,
            max_length=max_length,
            return_length=True,
            return_attention_mask=False,
            padding=False,
        )
        if fill_to_max_length:
            for i in range(len(self.tokens["input_ids"])):
                x = self.tokens["input_ids"][i]
                if max_length > len(x):
                    # Fill with the first non-pad token to the right.
                    self.tokens["input_ids"][i] = x + [x[0]] * (max_length - len(x))
                    self.tokens["length"][i] = max_length
        prompt_tokens = tokenizer(
            prompts,
            padding=False,
            truncation=True,
            return_length=True,
            max_length=max_length,
            return_attention_mask=False,
        )

        prompt_lengths = prompt_tokens["length"]
        seq_lengths = self.tokens["length"]
        prompt_masks = []
        for i in range(len(self)):
            prompt_len = prompt_lengths[i]
            seqlen = self.tokens["length"][i]
            # seq = self.tokens["input_ids"][i]
            # prompt = prompt_tokens["input_ids"][i]
            # assert seq[:prompt_len] == prompt, (seq, prompt, prompt_len, seqlen)
            assert seqlen >= prompt_len, (seqlen, prompt_len)
            prompt_mask = [1] * prompt_len + [0] * (seqlen - prompt_len)
            prompt_masks.append(prompt_mask)

        self.prompt_masks = prompt_masks

        logger.info(
            f"Loaded Prompt Answer Dataset with INFO: "
            f"#seqs={len(self)}, "
            f"truncation length={max_length}, "
            f"avg prompt length={np.mean(prompt_lengths):.1f}, "
            f"avg answer length={np.mean(seq_lengths) - np.mean(prompt_lengths):.1f}",
        )

    @property
    def util(self):
        return self._util

    def __len__(self):
        return len(self.tokens["input_ids"])

    def __getitem__(self, idx):
        d = {
            "packed_input_ids": torch.tensor(
                self.tokens["input_ids"][idx], dtype=torch.long
            ),
            "prompt_mask": torch.tensor(self.prompt_masks[idx], dtype=torch.bool),
        }
        assert len(d["packed_input_ids"]) == len(d["prompt_mask"])
        seqlen = [len(d["packed_input_ids"])]
        x = data_api.SequenceSample.from_default(
            ids=[self.ids[idx]],
            seqlens=seqlen,
            data=d,
        )
        return x


data_api.register_dataset("prompt_answer", PromptAnswerDataset)
