import time
from typing import List

from areal.extension.asystem.functioncall.base.call import (
    Language,
    batch_function_call,
    get_runtime_name,
)
from areal.extension.asystem.functioncall.base.utils import construct_uid, logger


async def ifeval_verify(
    id2info, generateds: List, query_ids: List, batch_size=1, timeout=1000
) -> List:
    assert len(generateds) == len(query_ids), (
        len(generateds),
        len(query_ids),
    )

    query_indices = []
    parameters = []
    for idx, (query_id, generated) in enumerate(zip(query_ids, generateds)):
        base_query_id = query_id.split("@idx:")[0]
        info = id2info[base_query_id]
        parameters.append(
            (
                generated,
                info["solutions"][0],
                idx,
            )
        )
        query_indices.append(idx)

    # Process in batches
    start_time = time.time()
    batch_args_list = []

    for i in range(0, len(parameters), batch_size):
        end_idx = min(i + batch_size, len(parameters))
        model_output_list, ground_truth_list, indices = zip(*parameters[i:end_idx])
        batch_args = {
            "model_output": list(model_output_list),
            "ground_truth": list(ground_truth_list),
            "query_ids": [query_ids[i] for i in indices],
        }

        sub_problem = {
            "uid": construct_uid(str(Language.INSTRUCT), i, end_idx),
            "language": str(Language.INSTRUCT).lower(),
            "runtime": get_runtime_name(None, str(Language.INSTRUCT).lower()),
            "code": 'print("hello ifeval!")',
            "testcases": [{}] * (end_idx - i),  # required filed
            "timeout": 10,
            "isFastFail": True,
            "extraInfo": batch_args,
        }

        batch_args_list.append(sub_problem)

    results_batch = await batch_function_call(
        batch_args_list, str(Language.INSTRUCT), timeout
    )

    labels = [0.0] * len(query_ids)
    # Map results back to original indices
    index = 0

    logger.warning(f"results_batch: {results_batch}.")

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
                f"Invalid functioncall ifeval results: {results}, batch index:{batch_idx}, query index: {query_indices[index]}, params: {batch_args_list[batch_idx]}."
            )
            continue

        for result in results["results"]:
            query_index = query_indices[index]
            labels[query_index] = float(result.get("stdout", 0.0))
            index += 1

    logger.info(
        f"verify ifeval with query size={len(query_ids)}, takes {time.time() - start_time:.4f} seconds, ifeval reward: {labels} results_batch:{results_batch}"
    )
    return labels


if __name__ == "__main__":
    # gt = r' {"input": "Answer with at least 17 words Add an introductory sentence for the following sentence.\nIt can be difficult to find that perfect gift.\n\nResponse:", "gold": "{"func_name": "validate_word_constraint", "N": 17, "quantifier": "at least", "end_phrase": null, "keyword_list": null, "word": null, "forbidden_words": null, "letter": null, "i": null, "first_word": null, "postscript_marker": null, "options": null, "section_splitter": null, "original_prompt": null}", "verifier": "ifeval-sample-verifier", "response": "Choosing the right present for someone can often be a challenging task, but it can be difficult to find that perfect gift.", "pass": "true"}'

    gt = '{"input":"Answer with at least 17 words Add an introductory sentence for the following sentence.\\nIt can be difficult to find that perfect gift.\\n\\nResponse:","gold":"{\\"func_name\\": \\"validate_word_constraint\\", \\"N\\": 17, \\"quantifier\\": \\"at least\\", \\"end_phrase\\": null, \\"keyword_list\\": null, \\"word\\": null, \\"forbidden_words\\": null, \\"letter\\": null, \\"i\\": null, \\"first_word\\": null, \\"postscript_marker\\": null, \\"options\\": null, \\"section_splitter\\": null, \\"original_prompt\\": null}","verifier":"ifeval-sample-verifier","response":"Choosing the right present for someone can often be a challenging task, but it can be difficult to find that perfect gift.","pass":"True"}'
    answer = "Choosing the right present for someone can often be a challenging task, but it can be difficult to find that perfect gift."

    # Let's test with both a correct and an incorrect answer
    correct_answer = answer
    incorrect_answer = r"\boxed{1 1 1...}"  # A deliberately wrong answer

    id2info = {
        "fe11b471-1aa9-4867-958f-a0a811c85f92": {
            "solutions": [gt],
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
    results = ifeval_verify(
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
