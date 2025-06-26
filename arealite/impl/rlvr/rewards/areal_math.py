# Copyright 2025 Ant Group Inc.
# Licensed under the Apache License, Version 2.0

from functools import lru_cache
from typing import List

from realhf.impl.dataset.math_code_dataset import load_metadata
from realhf.impl.dataset.math_parser import parse_line


@lru_cache(maxsize=1)
def _load_metadata(dataset_path: str):
    """Cached version of load_metadata to avoid reloading metadata each time."""
    return load_metadata(dataset_path)


def get_math_reward_fn(dataset_path):
    def math_reward(
        query_id: str,
        prompt: str,
        completion: str,
        prompt_ids: List[int],
        completion_ids: List[int],
        **kwargs,
    ) -> float:
        id2info, _ = _load_metadata(dataset_path)
        return parse_line(id2info=id2info, generated=completion, query_id=query_id)

    return math_reward
