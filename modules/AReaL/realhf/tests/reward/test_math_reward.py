import json
import os
from pathlib import Path

import numpy as np
import pytest

from realhf.impl.dataset.math_parser import verify_math_solution


def test_verify_math_solution():
    # The generated file is too large. Only upload sampled cases to git.
    path = Path("/storage/testing/dataset/math_generated.jsonl")
    line_numbers = np.random.choice(int(1e4), 10)
    if not os.path.exists(path):
        path = Path(__file__).parent / "math_answers_sample_cases.jsonl"
        line_numbers = list(range(10))
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if i not in line_numbers:
                continue
            line = json.loads(line)
            for ans, r in zip(line["generateds"], line["rewards"]):
                label = 0
                for sol in line["solutions"]:
                    label = label or verify_math_solution(ans, sol)
                assert (label - 0.5) * 10 == r
