import json
import os
import random
from collections import defaultdict
from datetime import datetime
from functioncall.base.utils import logger, constants
from functioncall.base.call import batch_function_call, Language, get_runtime_name


def construct_uid(query_id: str, start_idx: int, end_idx: int):
    try:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        trial_time = (
            f"{constants.experiment_name()}-{constants.trial_name()}-{timestamp}"
        )
    except Exception as e:
        trial_time = "test"
    uid = f"{trial_time}-{query_id}-case-{start_idx}-{end_idx}"
    return uid


def construct_testcases(
    inputs: list, outputs: list, index: tuple, remote: bool = False, is_ut: bool = False
) -> dict:
    result = []
    if is_ut:
        return result

    for i in range(*index):
        input, output = inputs[i].strip(), outputs[i].strip()
        if not remote:
            result.append({"input": input, "expectedOutput": output})
            continue

        oss_basepath = "https://antsys-hcsfaas-images-dev.cn-heyuan-alipay-office.oss-alipay.aliyuncs.com/areal/datasets/loj_0331"
        input_url = (
            input if input.startswith("http") else os.path.join(oss_basepath, input)
        )
        output_url = (
            output if input.startswith("http") else os.path.join(oss_basepath, output)
        )

        result.append({"input": input_url, "expectedOutput": output_url})
    return result


def load_problems_with_testcase_batch(
    id2info, query_ids, generateds, timeout_for_testcase, test_case_batch_size=1
):
    problem_list = []
    for idx, query_id in enumerate(query_ids):
        problem = id2info[query_id]
        # parse one problem
        language = problem.get("language", "PYTHON").upper()
        input_output = json.loads(problem["input_output"])
        fn_name = input_output.get("fn_name", "")
        remote = input_output.get("remote", False)
        inputs = input_output.get("inputs", [])
        outputs = input_output.get("outputs", [])
        assert len(inputs) == len(
            outputs
        ), f"Inputs({len(inputs)}) and outputs({len(outputs)}) mismatch for {query_id}"

        assert (
            language in Language.__members__
        ), f"{language} is not a valid Language name"

        is_ut = len(inputs) == 0

        # python + non-ut will choose fastFail,
        isFastFail = language == Language.PYTHON and not is_ut

        # create batches for testcases
        case_size = 1 if is_ut else len(inputs)
        test_case_batch_size = min(max(1, test_case_batch_size), case_size)

        for batch_idx in range(0, case_size, test_case_batch_size):
            end_idx = min(case_size, batch_idx + test_case_batch_size)
            testcases = construct_testcases(
                inputs, outputs, (batch_idx, end_idx), remote, is_ut
            )

            sub_problem = {
                "uid": construct_uid(query_id, batch_idx, end_idx),
                "language": language,
                "runtime": get_runtime_name("", language),
                "code": generateds[idx],
                "entryFunction": fn_name,
                "isFastFail": isFastFail,
                "isRemote": remote,
                "testcases": testcases,
                "timeout": timeout_for_testcase,
                "query_index": idx,
            }
            problem_list.append(sub_problem)

    return problem_list


def code_verify(
    id2info, generateds, query_ids, debug=False, timeout=1000, timeout_for_testcase=6
):
    assert len(generateds) == len(query_ids), (
        len(generateds),
        len(query_ids),
    )
    payload_list = []

    payload_list = load_problems_with_testcase_batch(
        id2info,
        query_ids,
        generateds,
        timeout_for_testcase,
        test_case_batch_size=20,
    )

    logger.info(
        f"code_verify start, request count: {len(payload_list)}, query size: {len(query_ids)}, query_id_0: {query_ids[0]}"
    )
    rsp_list = batch_function_call(payload_list, "code", timeout)

    results = [1] * len(query_ids) if len(rsp_list) else [0] * len(query_ids)
    for idx, rsp in enumerate(rsp_list):
        query_index = payload_list[idx]["query_index"]
        query_id = query_ids[query_index]

        value = 0
        if rsp and rsp.get("success", False):
            value = 1
        else:
            print(
                f'Functioncall code verify not passed, uid: {rsp.get("uid")}, query id: {query_id}, results: {rsp}'
            )

        results[query_index] = results[query_index] and value

    logger.info(
        f"code_verify finished, request count: {len(payload_list)}, query count: {len(query_ids)}, result count: {len(results)}"
    )
    return results


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


if __name__ == "__main__":
    # data_list = [
    #     {
    #         "task": "code",
    #         "query_id": "",
    #         "prompt": "",
    #         "solutions": [
    #             'from operator import *\n\nfrom typing import *\n\nclass Solution:\n    def solveNQueens(self, n: int) -> List[List[str]]:\n        def generateBoard():\n            board = list()\n            for i in range(n):\n                row[queens[i]] = "Q"\n                board.append("".join(row))\n                row[queens[i]] = "."\n            return board\n\n        def solve(row: int, columns: int, diagonals1: int, diagonals2: int):\n            if row == n:\n                board = generateBoard()\n                solutions.append(board)\n            else:\n                availablePositions = ((1 << n) - 1) & (~(columns | diagonals1 | diagonals2))\n                while availablePositions:\n                    position = availablePositions & (-availablePositions)\n                    availablePositions = availablePositions & (availablePositions - 1)\n                    column = bin(position - 1).count("1")\n                    queens[row] = column\n                    solve(row + 1, columns | position, (diagonals1 | position) << 1, (diagonals2 | position) >> 1)\n\n        solutions = list()\n        queens = [-1] * n\n        row = ["."] * n\n        solve(0, 0, 0, 0)\n        return solutions\n# Test case 1: Smallest case, n = 1\n# There is only one queen, so the only solution is a board with a single \'Q\'.\nsolution = Solution()\nassert solution.solveNQueens(1) == [[\'Q\']]\n'
    #         ],
    #         "input_output": '{"inputs":[],"outputs":[],"fn_name":"","remote":false}',
    #         "language": "PYTHON",
    #     }
    # ]

    data_list = load_jsonl("functioncall/test/test_dataset.jsonl")
    id2info = defaultdict(dict)
    for item in data_list:
        query_id = str(item["query_id"])
        id2info[query_id] = item
        # id2info[query_id]["input_output"]["remote"] = True
        # id2info[query_id]["language"] = "PYTHON"

    def create_test_params(count=10):
        query_ids = []
        generateds = []
        cnt = 0

        for d in data_list:
            if cnt >= count:
                break
            if not d["solutions"] or d["query_id"] not in id2info:
                continue
            query_ids.append(d["query_id"])
            generateds.extend(d["solutions"])
            cnt += 1

        return generateds, query_ids

    generateds, query_ids = create_test_params(10)
    # generateds, query_ids = ["s = input()\nprint(s)\n"], ["loj_6053"]
    print(f"generateds:, query_ids:{query_ids}")
    result = code_verify(id2info, generateds, query_ids, True)
    print(result)


if __name__ == "__main__1":
    data_list = load_jsonl(
        "/Users/jun/Documents/code/AReaL/functioncall/loj_6053/loj_code.jsonl"
    )
    id2info = defaultdict(dict)
    for item in data_list:
        query_id = str(item["id"])
        id2info[query_id] = item
        # id2info[query_id]["input_output"]["remote"] = True
        id2info[query_id]["language"] = "PYTHON"

    def create_test_params(count=10):
        query_ids = []
        generateds = []
        cnt = 0

        file_path = "/storage/openpsi/users/meijun.mei/datasets/Scenario.codegeneration_10_0.2_eval_all.json"
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

    # generateds, query_ids = create_test_params(10)

    generateds, query_ids = ["s = input()\nprint(s)\n"], ["loj_6053"]
    print(f"generateds:, query_ids:{query_ids}")
    result = code_verify(id2info, generateds, query_ids, True)
    print(result)
