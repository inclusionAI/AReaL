import json
import os
import time
from typing import List

from functioncall.base.call import batch_function_call
from realhf.base import logging

logger = logging.getLogger("function call")


def math_verify(
    id2info, generateds: List, query_ids: List, batch_size=10, timeout=1000
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
        for cur_solution in info["solutions"]:
            parameters.append((generated, cur_solution, idx))
            query_indices.append(idx)

    # Process in batches
    start_time = time.time()
    batch_args_list = []
    for i in range(0, len(parameters), batch_size):
        answers, solutions, indices = zip(*parameters[i : i + batch_size])
        batch_args = {
            "answers": list(answers),
            "solutions": list(solutions),
            "query_ids": [query_ids[i] for i in indices],
        }

        batch_args_list.append(batch_args)

    results_batch = batch_function_call(batch_args_list, "python_math", timeout)

    labels = [0] * len(query_ids)
    # Map results back to original indices
    index = 0
    for batch_idx, results in enumerate(results_batch):
        if not isinstance(results, list) or len(results) == 0:
            index += len(batch_args_list[batch_idx]["answers"])
            logger.warning(
                f"Invalid functioncall math results: {results}, batch index:{batch_idx}, query index: {query_indices[index]}, params: {batch_args_list[batch_idx]['answers']}."
            )
            continue

        for result in results:
            query_index = query_indices[index]
            if (
                isinstance(result, list)
                and len(result) > 0
                and (isinstance(result[0], int) and result[0] in [0, 1])
            ):
                labels[query_index] = result[0] or labels[query_index]
            else:
                logger.warning(
                    f"Invalid functioncall math result: {result}, index:{index}, qeury_id: {query_ids[query_index]}."
                )

            index += 1

    logger.info(
        f"verify math with query size={len(query_ids)}, takes {time.time() - start_time:.4f} seconds"
    )
    return labels


if __name__ == "__main__":
    sample = {
        "answers": ["-\\frac{2}{3}"],
        "solutions": [
            "1. **Apply the operation $\\otimes$ to the innermost parentheses first:**\n   \\[\n   (1 \\otimes 2) \\otimes 3 = \\left(\\frac{1^2}{2}\\right) \\otimes 3 = \\frac{1}{2} \\otimes 3\n   \\]\n   \\[\n   1 \\otimes (2 \\otimes 3) = 1 \\otimes \\left(\\frac{2^2}{3}\\right) = 1 \\otimes \\frac{4}{3}\n   \\]\n\n2. **Calculate each part using the definition of $\\otimes$:**\n   \\[\n   \\frac{1}{2} \\otimes 3 = \\frac{\\left(\\frac{1}{2}\\right)^2}{3} = \\frac{\\frac{1}{4}}{3} = \\frac{1}{12}\n   \\]\n   \\[\n   1 \\otimes \\frac{4}{3} = \\frac{1^2}{\\frac{4}{3}} = \\frac{1}{\\frac{4}{3}} = \\frac{3}{4}\n   \\]\n\n3. **Subtract the two results:**\n   \\[\n   \\left(\\frac{1}{12}\\right) - \\left(\\frac{3}{4}\\right) = \\frac{1}{12} - \\frac{9}{12} = -\\frac{8}{12} = -\\frac{2}{3}\n   \\]\n\n4. **Conclude with the final answer:**\n   \\[\n   \\boxed{A}\n   \\]",
            "\\boxed{-\\frac{2}{3}}",
        ],
    }
    id2info = {"fe11b471-1aa9-4867-958f-a0a811c85f92": sample}

    start_time = time.time()
    result = math_verify(
        id2info,
        sample["answers"] * 100,
        ["fe11b471-1aa9-4867-958f-a0a811c85f92" for _ in range(100)],
    )
    print(result)
