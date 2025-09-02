import os
import time
from typing import List

from arealite.extension.asystem.functioncall.base.call import (
    Language,
    batch_function_call,
    get_runtime_name,
)
from arealite.extension.asystem.functioncall.base.utils import construct_uid, logger


async def logic_verify(
    id2info, generateds: List, query_ids: List, batch_size=1, timeout=1000
) -> List:
    assert len(generateds) == len(query_ids), (
        len(generateds),
        len(query_ids),
    )

    st = time.monotonic()
    query_indices = []
    parameters = []
    # Collect all (generated, solution) pairs with their original indices
    for idx, (query_id, generated) in enumerate(zip(query_ids, generateds)):
        base_query_id = query_id.split("@idx:")[0]
        info = id2info[base_query_id]
        parameters.append(
            (
                generated,
                info["solutions"][0],
                info["puzzle"],
                info["puzzle_type"],
                idx,
            )
        )
        query_indices.append(idx)

    # Process in batches
    start_time = time.time()
    batch_args_list = []

    for i in range(0, len(parameters), batch_size):
        end_idx = min(i + batch_size, len(parameters))
        solution_str_list, ground_truth_list, puzzle_list, puzzle_type_list, indices = (
            zip(*parameters[i:end_idx])
        )
        batch_args = {
            "solution_str": list(solution_str_list),
            "ground_truth": list(ground_truth_list),
            "puzzle": list(puzzle_list),
            "puzzle_type": list(puzzle_type_list),
            "query_ids": [query_ids[i] for i in indices],
        }

        sub_problem = {
            "uid": construct_uid(str(Language.LOGIC), i, end_idx),
            "language": str(Language.LOGIC).lower(),
            "runtime": get_runtime_name(None, str(Language.LOGIC).lower()),
            "code": 'print("hello logic!")',
            "testcases": [{}] * (end_idx - i),  # required filed
            "timeout": 10,
            "isFastFail": True,
            "extraInfo": batch_args,
        }

        batch_args_list.append(sub_problem)

    results_batch = await batch_function_call(batch_args_list, str(Language.LOGIC), timeout)

    labels = [0.0] * len(query_ids)
    # Map results back to original indices
    index = 0

    for batch_idx, results in enumerate(results_batch):
        # check result format
        if not (
            isinstance(results, dict)
            and "results" in results
            and isinstance(results["results"], list)
            and results["results"]
            and all(
                isinstance(item, dict) and "stdout" in item
                for item in results["results"]
            )
        ):
            index += len(batch_args_list[batch_idx]["extraInfo"]["query_ids"])
            logger.warning(
                f"Invalid functioncall logic results: {results}, batch index:{batch_idx}, query index: {query_indices[index]}, params: {batch_args_list[batch_idx]}."
            )
            continue

        for result in results["results"]:
            query_index = query_indices[index]
            # set label as 1 if any of the solutions matches the answer
            labels[query_index] = float(result.get("stdout", 0.0))
            index += 1

    logger.info(
        f"verify logic with query size={len(query_ids)}, takes {time.time() - start_time:.4f} seconds, logic reward: {labels} results_batch:{results_batch}"
    )
    return labels


if __name__ == "__main__":
    puzzle = '{"train": [{"input": [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 5, 5, 5, 0, 0, 0, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0], [0, 5, 5, 5, 0, 0, 0, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0], [0, 5, 5, 5, 0, 0, 0, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 5, 5, 5, 0, 0, 0, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0], [0, 5, 5, 5, 0, 0, 0, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0], [0, 5, 5, 5, 0, 0, 0, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], "output": [[5, 0, 5, 0, 0, 0, 5, 0, 5], [0, 5, 0, 0, 0, 0, 0, 5, 0], [5, 0, 5, 0, 0, 0, 5, 0, 5], [0, 0, 0, 5, 0, 5, 0, 0, 0], [0, 0, 0, 0, 5, 0, 0, 0, 0], [0, 0, 0, 5, 0, 5, 0, 0, 0], [5, 0, 5, 0, 0, 0, 5, 0, 5], [0, 5, 0, 0, 0, 0, 0, 5, 0], [5, 0, 5, 0, 0, 0, 5, 0, 5]]}, {"input": [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 0, 0], [0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], "output": [[5, 5, 0, 5, 5, 0, 0, 0, 0], [0, 0, 5, 0, 0, 5, 0, 0, 0], [5, 5, 0, 5, 5, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 5, 5, 0], [0, 0, 0, 0, 0, 0, 0, 0, 5], [0, 0, 0, 0, 0, 0, 5, 5, 0], [5, 5, 0, 5, 5, 0, 0, 0, 0], [0, 0, 5, 0, 0, 5, 0, 0, 0], [5, 5, 0, 5, 5, 0, 0, 0, 0]]}, {"input": [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0], [0, 0, 0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0], [0, 0, 0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0], [0, 0, 0, 5, 5, 5, 0, 0, 0, 5, 5, 5, 0, 0, 0, 0, 0, 0], [0, 0, 0, 5, 5, 5, 0, 0, 0, 5, 5, 5, 0, 0, 0, 0, 0, 0], [0, 0, 0, 5, 5, 5, 0, 0, 0, 5, 5, 5, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], "output": [[5, 5, 5, 5, 5, 5, 5, 5, 5], [0, 5, 5, 0, 5, 5, 0, 5, 5], [5, 0, 5, 5, 0, 5, 5, 0, 5], [0, 0, 0, 5, 5, 5, 5, 5, 5], [0, 0, 0, 0, 5, 5, 0, 5, 5], [0, 0, 0, 5, 0, 5, 5, 0, 5], [5, 5, 5, 0, 0, 0, 5, 5, 5], [0, 5, 5, 0, 0, 0, 0, 5, 5], [5, 0, 5, 0, 0, 0, 5, 0, 5]]}], "test": [{"input": [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 0, 0, 0, 5, 5, 5, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 0, 0, 0, 5, 5, 5, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 0, 0, 0, 5, 5, 5, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], "output": [[5, 5, 5, 5, 5, 5, 5, 5, 5], [0, 5, 0, 0, 5, 0, 0, 5, 0], [5, 0, 5, 5, 0, 5, 5, 0, 5], [0, 0, 0, 5, 5, 5, 0, 0, 0], [0, 0, 0, 0, 5, 0, 0, 0, 0], [0, 0, 0, 5, 0, 5, 0, 0, 0], [5, 5, 5, 0, 0, 0, 5, 5, 5], [0, 5, 0, 0, 0, 0, 0, 5, 0], [5, 0, 5, 0, 0, 0, 5, 0, 5]]}]}'
    puzzle_type = "ARC-AGI"
    gt = r"\boxed{5 5 5 5 5 5 5 5 5 \n 0 5 0 0 5 0 0 5 0 \n 5 0 5 5 0 5 5 0 5 \n 0 0 0 5 5 5 0 0 0 \n 0 0 0 0 5 0 0 0 0 \n 0 0 0 5 0 5 0 0 0 \n 5 5 5 0 0 0 5 5 5 \n 0 5 0 0 0 0 0 5 0 \n 5 0 5 0 0 0 5 0 5 \n}"
    answer = r"\boxed{5 5 5 5 5 5 5 5 5 \n 0 5 0 0 5 0 0 5 0 \n 5 0 5 5 0 5 5 0 5 \n 0 0 0 5 5 5 0 0 0 \n 0 0 0 0 5 0 0 0 0 \n 0 0 0 5 0 5 0 0 0 \n 5 5 5 0 0 0 5 5 5 \n 0 5 0 0 0 0 0 5 0 \n 5 0 5 0 0 0 5 0 5 \n}"

    # Let's test with both a correct and an incorrect answer
    correct_answer = answer
    incorrect_answer = r"\boxed{1 1 1...}"  # A deliberately wrong answer

    id2info = {
        "fe11b471-1aa9-4867-958f-a0a811c85f92": {
            "solutions": [gt],
            "puzzle": puzzle,
            "puzzle_type": puzzle_type,
        }
    }

    num_items = 10
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
    actual_correct_count = sum(int(result) for result in results)
    print(
        f"\nTest Finished. Expected correct count: {expected_correct_count}, Actual correct count: {actual_correct_count}, results: {results}"
    )
    assert expected_correct_count == actual_correct_count
    print(f"Test Passed! cost time: {time.time() - start_time}")
