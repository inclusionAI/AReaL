import json
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from areal.extension.asystem.functioncall.base.utils import logger

# pip install logic-verifier==0.0.2 -i https://pypi.antfin-inc.com/simple-dev/


def compute_score(
    solution_str, ground_truth, puzzle, puzzle_type, format_score=0, dynamic_param={}
) -> float:
    """
    Computes the score for a single solution against its ground truth.
    This function encapsulates the local verification logic.
    """
    from verifier.logic_puzzle_verify import LogicPuzzleVerifier

    score = 0
    try:
        verify_result_str = LogicPuzzleVerifier().verify(
            puzzle, puzzle_type, solution_str, ground_truth, format_score, dynamic_param
        )
        verify_result = json.loads(verify_result_str)

        if "is_format_correct" not in verify_result:
            raise Exception(f"Invalid verification result format: {verify_result_str}")

        is_format_correct = verify_result["is_format_correct"]
        if is_format_correct == 1:
            score = format_score

        precision = verify_result.get("precision", 0.0)
        if precision != 0:
            score += precision

        return float(score)
    except Exception as e:
        return score


def logic_verify(
    id2info: Dict[str, Any],
    generateds: List[str],
    query_ids: List[str],
    num_threads: int = os.cpu_count() or 8,
) -> List[int]:
    if not generateds:
        return []

    assert len(generateds) == len(
        query_ids
    ), f"Mismatch in input lengths: {len(generateds)} generations vs {len(query_ids)} query IDs."

    logger.info(
        f"Starting local logic verification for {len(generateds)} items using {num_threads} threads..."
    )
    start_time = time.time()

    tasks_args = []
    for i, (query_id, generated) in enumerate(zip(query_ids, generateds)):
        base_query_id = query_id.split("@idx:")[0]
        info = id2info[base_query_id]

        args = (
            generated,  # solution_str
            info["solutions"][0],  # ground_truth (assuming one ground truth solution)
            info["puzzle"],  # puzzle
            info["puzzle_type"],  # puzzle_type
        )
        tasks_args.append(args)

    results = [0.0] * len(generateds)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_index = {
            executor.submit(compute_score, *args): i
            for i, args in enumerate(tasks_args)
        }

        for future in as_completed(future_to_index):
            original_index = future_to_index[future]
            try:
                score = future.result()
                results[original_index] = score
            except Exception as exc:
                logger.error(
                    f"An exception occurred for item at index {original_index}: {exc}"
                )
                results[original_index] = 0.0

    labels = [int(score > 0) for score in results]

    logger.info(
        f"Finished verification. Total time: {time.time() - start_time:.4f} seconds. "
        f"Correct count: {sum(labels)}/{len(labels)}."
    )
    return labels


if __name__ == "__main__":
    puzzle = '{"train": [{"input": [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 5, 5, 5, 0, 0, 0, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0], [0, 5, 5, 5, 0, 0, 0, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0], [0, 5, 5, 5, 0, 0, 0, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 5, 5, 5, 0, 0, 0, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0], [0, 5, 5, 5, 0, 0, 0, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0], [0, 5, 5, 5, 0, 0, 0, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], "output": [[5, 0, 5, 0, 0, 0, 5, 0, 5], [0, 5, 0, 0, 0, 0, 0, 5, 0], [5, 0, 5, 0, 0, 0, 5, 0, 5], [0, 0, 0, 5, 0, 5, 0, 0, 0], [0, 0, 0, 0, 5, 0, 0, 0, 0], [0, 0, 0, 5, 0, 5, 0, 0, 0], [5, 0, 5, 0, 0, 0, 5, 0, 5], [0, 5, 0, 0, 0, 0, 0, 5, 0], [5, 0, 5, 0, 0, 0, 5, 0, 5]]}, {"input": [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 0, 0], [0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], "output": [[5, 5, 0, 5, 5, 0, 0, 0, 0], [0, 0, 5, 0, 0, 5, 0, 0, 0], [5, 5, 0, 5, 5, 0, 0, 0, 0], [0, 0, 0, 0, 0 , 0, 5, 5, 0], [0, 0, 0, 0, 0, 0, 0, 0, 5], [0, 0, 0, 0, 0, 0, 5, 5, 0], [5, 5, 0, 5, 5, 0, 0, 0, 0], [0, 0, 5, 0, 0, 5, 0, 0, 0], [5, 5, 0, 5, 5, 0, 0, 0, 0]]}, {"input": [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0], [0, 0, 0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0], [0, 0, 0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0], [0, 0, 0, 5, 5, 5, 0, 0, 0, 5, 5, 5, 0, 0, 0, 0, 0, 0], [0, 0, 0, 5, 5, 5, 0, 0, 0, 5, 5, 5, 0, 0, 0, 0, 0, 0], [0, 0, 0, 5, 5, 5, 0, 0, 0, 5, 5, 5, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], "output": [[5, 5, 5, 5, 5, 5, 5, 5, 5], [0, 5, 5, 0, 5, 5, 0, 5, 5], [5, 0, 5, 5, 0, 5, 5, 0, 5], [0, 0, 0, 5, 5, 5, 5, 5, 5], [0, 0, 0, 0, 5, 5, 0, 5, 5], [0, 0, 0, 5, 0, 5, 5, 0, 5], [5, 5, 5, 0, 0, 0, 5, 5, 5], [0, 5, 5, 0, 0, 0, 0, 5, 5], [5, 0, 5, 0, 0, 0, 5, 0, 5]]}], "test": [{"input": [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 0, 0, 0, 5, 5, 5, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 0, 0, 0, 5, 5, 5, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 0, 0, 0, 5, 5, 5, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], "output": [[5, 5, 5, 5, 5, 5, 5, 5, 5], [0, 5, 0, 0, 5, 0, 0, 5, 0], [5, 0, 5, 5, 0, 5, 5, 0, 5], [0, 0, 0, 5, 5, 5, 0, 0, 0], [0, 0, 0, 0, 5, 0, 0, 0, 0], [0, 0, 0, 5, 0, 5, 0, 0, 0], [5, 5, 5, 0, 0, 0, 5, 5, 5], [0, 5, 0, 0, 0, 0, 0, 5, 0], [5, 0, 5, 0, 0, 0, 5, 0, 5]]}]}'
    puzzle_type = "ARC-AGI"
    gt = r"\boxed{5 5 5 5 5 5 5 5 5 \n 0 5 0 0 5 0 0 5 0 \n 5 0 5 5 0 5 5 0 5 \n 0 0 0 5 5 5 0 0 0 \n 0 0 0 0 5 0 0 0 0 \n 0 0 0 5 0 5 0 0 0 \n 5 5 5 0 0 0 5 5 5 \n 0 5 0 0 0 0 0 5 0 \n 5 0 5 0 0 0 5 0 5 \n}"

    # Let's test with both a correct and an incorrect answer
    correct_answer = gt
    incorrect_answer = r"\boxed{1 1 1...}"  # A deliberately wrong answer

    id2info = {
        "fe11b471-1aa9-4867-958f-a0a811c85f92": {
            "solutions": [gt],
            "puzzle": puzzle,
            "puzzle_type": puzzle_type,
        }
    }

    num_items = 1000
    all_generateds = []
    all_query_ids = []
    for i in range(num_items):
        if i % 2 == 0:
            all_generateds.append(
                correct_answer
            )  # Every even index is a correct answer
        else:
            all_generateds.append(
                incorrect_answer
            )  # Every odd index is an incorrect answer
        all_query_ids.append("fe11b471-1aa9-4867-958f-a0a811c85f92")

    print(f"\n--- Testing with {num_items} items ---")

    start_time = time.time()
    results = logic_verify(
        id2info,
        all_generateds,
        all_query_ids,
    )

    # Verify the results
    expected_correct_count = num_items // 2
    actual_correct_count = sum(results)
    print(
        f"\nTest Finished. Expected correct count: {expected_correct_count}, Actual correct count: {actual_correct_count}, results: {results}"
    )
    assert expected_correct_count == actual_correct_count
    print(f"Test Passed! cost time: {time.time() - start_time}")
