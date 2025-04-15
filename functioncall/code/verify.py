import json
import os
import random
from collections import defaultdict

from functioncall.base.call import batch_function_call
from realhf.base import logging

logger = logging.getLogger("function call")


def load_problems_with_testcase_batch(id2info, query_ids, test_case_batch_size=None):
    problem_map = defaultdict(list)
    for idx, query_id in enumerate(query_ids):
        problem = id2info[query_id]
        # parse one problem
        input_output = json.loads(problem["input_output"])
        inputs = input_output.get("inputs", [])
        outputs = input_output.get("outputs", [])
        assert len(inputs) == len(
            outputs
        ), f"Inputs({len(inputs)}) and outputs({len(outputs)}) mismatch for {query_id}"

        other_io_fields = {
            k: v for k, v in input_output.items() if k not in ["inputs", "outputs"]
        }
        # create batches for testcases
        if not test_case_batch_size or test_case_batch_size <= 0:
            test_case_batch_size = len(inputs)

        for batch_idx in range(0, len(inputs), test_case_batch_size):
            batch_io = {
                **other_io_fields,
                "inputs": inputs[batch_idx : batch_idx + test_case_batch_size],
                "outputs": outputs[batch_idx : batch_idx + test_case_batch_size],
            }

            sub_problem = {
                "problem_id": query_id,
                "input_output": json.dumps(batch_io),
                "batche_index": batch_idx,
            }
            problem_map[query_id].append(sub_problem)

    return problem_map


def code_verify(
    id2info, generateds, query_ids, debug=False, timeout=1000, timeout_for_testcase=6
):
    assert len(generateds) == len(query_ids), (
        len(generateds),
        len(query_ids),
    )
    payload_list = []

    global_problems = load_problems_with_testcase_batch(
        id2info,
        query_ids,
        test_case_batch_size=20,
    )
    for idx, query_id in enumerate(query_ids):
        problems = global_problems[query_id]
        for problem in problems:
            payload_list.append(
                {
                    "problem": problem,
                    "code": generateds[idx],
                    "debug": debug,
                    "timeout": timeout_for_testcase,
                    "query_index": idx,
                }
            )

    logger.info(
        f"code_verify start, payload_list size: {len(payload_list)}, query size: {len(query_ids)}, query_id_0: {query_ids[0]}"
    )
    rsp_list = batch_function_call(payload_list, "python_code", timeout=timeout)

    results = [1] * len(query_ids)
    for idx, rsp in enumerate(rsp_list):
        query_index = payload_list[idx]["query_index"]
        query_id = query_ids[query_index]

        value = 0
        if rsp and "result" in rsp and not any(x != True for x in rsp["result"]):
            value = 1
        else:
            logger.debug(
                f"Functioncall code verify not passed, query index: {query_index}, query id: {query_id}, results: {rsp}"
            )

        results[query_index] = results[query_index] and value

    logger.info(
        f"code_verify finished, payload_list size: {len(payload_list)}, query size: {len(query_ids)}, result size: {len(results)}"
    )
    return results


if __name__ == "__main__":

    def load_jsonl(file_path: str):
        """Load JSONL file with validation"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return [json.loads(line) for line in f]
        except FileNotFoundError:
            print(f"ERROR: JSONL file not found: {file_path}")
            raise
        except json.JSONDecodeError as e:
            print(f"ERROR: JSON parsing failed in {file_path}: {str(e)}")
            raise

    data4 = load_jsonl("input.jsonl")
    id2info = defaultdict(dict)
    for item in data4:
        query_id = str(item["query_id"])
        id2info[query_id] = item

    def create_test_params(count=10):
        query_ids = []
        generateds = []
        cnt = 0

        file_path = "lcb_code.json"
        raw_data = []
        with open(file_path, "r", encoding="utf-8") as f:
            raw_data = [line for line in json.load(f)]

        for d in raw_data:
            if cnt >= count:
                break
            if not d["code_list"] or d["question_id"] not in id2info:
                continue
            query_ids.append(d["question_id"])
            generateds.append(d["code_list"][0])
            cnt += 1

        return generateds, query_ids

    generateds, query_ids = create_test_params(10)
    print(f"generateds:, query_ids:{query_ids}")
    result = code_verify(id2info, generateds, query_ids, True)
    print(result)
