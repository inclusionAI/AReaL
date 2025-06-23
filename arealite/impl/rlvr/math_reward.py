# Copyright 2025 Ant Group Inc.
# Licensed under the Apache License, Version 2.0

from functools import lru_cache
from typing import List

from realhf.impl.dataset.math_code_dataset import load_metadata
from realhf.impl.dataset.math_parser import parse_lines_in_parallel


@lru_cache(maxsize=1)
def _load_metadata(dataset_path: str):
    """Cached version of load_metadata to avoid reloading metadata each time."""
    return load_metadata(dataset_path)


def get_math_reward_fn(dataset_path):
    def math_reward(
        query_ids: List[str],
        prompts: List[str],
        completions: List[str],
        prompt_ids: List[List[int]],
        completion_ids: List[List[int]],
    ) -> List[int]:
        id2info, _ = _load_metadata(dataset_path)
        for qid in query_ids:
            assert qid in id2info
            assert id2info[qid].get("task", "math") == "math"
        return parse_lines_in_parallel(
            id2info=id2info, generateds=completions, query_ids=query_ids
        )

    return math_reward
