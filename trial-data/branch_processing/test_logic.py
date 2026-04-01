"""
Dry-run test: validates the splitting and wrapping logic offline
(no sglang server needed).

Run:  python test_logic.py
"""

import json
import re
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from process_branches import split_by_alternatively, count_alternatively


def fake_process(assistant_content: str, num_branches: int = 2) -> str:
    """
    Same logic as process_assistant_content but uses a FAKE generator
    that returns '[GENERATED BRANCH]' so we can inspect the structure.
    """
    n = count_alternatively(assistant_content)
    if n <= 1:
        return assistant_content

    segments = split_by_alternatively(assistant_content)
    result_parts = [segments[0]]

    for i in range(1, n):
        original_block = "Alternatively" + segments[i]

        # Fake branches
        branches = [f"[GENERATED BRANCH {b+1} for Alt #{i}]" for b in range(num_branches)]

        trial_parts = [f"<Trial>{original_block}</Trial>"]
        for branch_text in branches:
            trial_parts.append(f"<Trial>{branch_text}</Trial>")

        parallel_block = "<Parallel>" + "".join(trial_parts) + "</Parallel>"
        result_parts.append(parallel_block)

    # Last Alternatively
    result_parts.append("Alternatively" + segments[n])

    return "".join(result_parts)


def test_basic():
    """Test with a simple example."""
    text = "Part A. Alternatively, Part B. Alternatively, Part C. Alternatively, Part D."
    n = count_alternatively(text)
    print(f"Number of 'Alternatively': {n}")
    assert n == 3, f"Expected 3, got {n}"

    segments = split_by_alternatively(text)
    print(f"Segments: {segments}")
    assert len(segments) == 4, f"Expected 4 segments, got {len(segments)}"

    result = fake_process(text, num_branches=2)
    print(f"\nProcessed text:\n{result}\n")

    # Verify structure
    assert "<Parallel>" in result
    assert "<Trial>" in result
    assert result.count("<Parallel>") == 2  # n-1 = 2 blocks
    assert result.startswith("Part A. ")
    assert result.endswith("Alternatively, Part D.")
    print("test_basic PASSED\n")


def test_single_alternatively():
    """With only 1 Alternatively, no branching should happen."""
    text = "Part A. Alternatively, Part B."
    result = fake_process(text)
    assert result == text, f"Expected no change, got: {result}"
    print("test_single_alternatively PASSED\n")


def test_no_alternatively():
    """With 0 Alternatively, no branching should happen."""
    text = "Just a normal response."
    result = fake_process(text)
    assert result == text
    print("test_no_alternatively PASSED\n")


def test_on_real_data():
    """Run fake_process on the first few real samples."""
    input_path = os.path.join(os.path.dirname(__file__), "..", "train.jsonl")
    if not os.path.exists(input_path):
        print(f"Skipping real data test: {input_path} not found")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        branched_count = 0
        for idx, line in enumerate(f):
            if idx >= 5:
                break
            data = json.loads(line)
            for msg in data["messages"]:
                if msg["role"] == "assistant":
                    content = msg["content"]
                    n = count_alternatively(content)
                    if n >= 2:
                        result = fake_process(content, num_branches=1)
                        # Count parallel blocks
                        n_parallel = result.count("<Parallel>")
                        print(f"Sample {idx}: n_alt={n}, n_parallel={n_parallel} (expected {n-1})")
                        assert n_parallel == n - 1
                        branched_count += 1
                    else:
                        print(f"Sample {idx}: n_alt={n} (no branching)")
                    break

        print(f"\ntest_on_real_data PASSED ({branched_count} samples branched)\n")


if __name__ == "__main__":
    test_basic()
    test_single_alternatively()
    test_no_alternatively()
    test_on_real_data()
    print("ALL TESTS PASSED")
