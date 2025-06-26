# Copyright 2025 Ant Group Inc.
# Licensed under the Apache License, Version 2.0

import re
from functools import lru_cache
from typing import List

from functioncall.code.local_verify import code_verify
from realhf.impl.dataset.math_code_dataset import load_metadata


@lru_cache(maxsize=1)
def _load_metadata(dataset_path: str):
    """Cached version of load_metadata to avoid reloading metadata each time."""
    return load_metadata(dataset_path)


def extract_code(text, min_length=20):
    """Extract code blocks from text."""
    code_pattern = r"(?i)```(?:python|py|cpp|CPP)?\s*\n?(.*?)\n?```"
    code_blocks = re.findall(code_pattern, text, re.DOTALL)
    valid_blocks = []
    for block in code_blocks:
        clean_block = block.strip()
        if len(clean_block) < min_length:
            continue
        valid_blocks.append(clean_block)

    if not valid_blocks:
        return None
    # return the last code block
    return valid_blocks[-1]


def get_code_reward_fn(dataset_path):
    def code_reward(
        query_id: str,
        prompt: str,
        completion: str,
        prompt_ids: List[int],
        completion_ids: List[int],
        **kwargs,
    ) -> float:
        id2info, _ = _load_metadata(dataset_path)
        return code_verify(
            id2info=id2info, generateds=[extract_code(completion)], query_ids=[query_id]
        )[0]

    return code_reward
