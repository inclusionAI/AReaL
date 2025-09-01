import json
import os
from collections import defaultdict
from typing import Any, Dict, Optional

from arealite.extension.asystem.functioncall.base.call import (
    Language,
    batch_function_call,
    get_runtime_name,
)
from arealite.extension.asystem.functioncall.base.utils import (
    construct_uid,
    load_jsonl,
    logger,
)

SINGLE_CASE_EXEC_TIMEOUT = 6
TEST_CASE_BATCH_SIZE = 1
FUNCTIONCALL_TIMEOUT = 1000


def round_up_memory(memory):
    if memory <= 0:
        return 0
    rounded = int(((memory + 255) // 256) * 256)
    return 0 if rounded > 1024 else rounded


def round_up_timeout(timeout):
    timeout = float(timeout)
    if timeout <= 0:
        return SINGLE_CASE_EXEC_TIMEOUT
    return min(12, max(0.1, timeout * 1.5))  # [0.1, 12] s


def construct_testcases(
    inputs: list, outputs: list, index: tuple, remote: bool = False, is_ut: bool = False
) -> dict:
    result = []
    if is_ut:
        return result

    for i in range(*index):
        print(
            f"[inputs: {inputs[i]} - {inputs[i].strip()} - {outputs[i]}-{outputs[i].strip()}]"
        )
        input_, output_ = inputs[i], outputs[i]  # inputs[i].strip(), outputs[i].strip()
        if not remote:
            result.append({"input": input_, "expectedOutput": output_})
            continue

        oss_basepath = "http://antsys-hcsfaas-images-dev.cn-heyuan-alipay-office.oss-alipay.aliyuncs.com/"
        input_url = (
            input_ if input_.startswith("http") else os.path.join(oss_basepath, input_)
        )
        output_url = (
            output_
            if output_.startswith("http")
            else os.path.join(oss_basepath, output_)
        )

        result.append({"input": input_url, "expectedOutput": output_url})
    return result


def load_problems_with_testcase_batch(
    id2info, query_ids, generateds, timeout_for_testcase, test_case_batch_size
):
    problem_list = []
    for idx, query_id in enumerate(query_ids):
        problem = id2info[query_id]
        # parse one problem
        language = problem.get("language", "PYTHON").upper()
        timeout = round_up_timeout(problem.get("timeout", 0))
        memory = round_up_memory(problem.get("memory", 0))
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

        # isFastFail means the function call returns immediately as soon as any testcase fails.
        isFastFail = True
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
                "timeout": timeout,
                "memory": memory,
                "query_index": idx,
            }
            problem_list.append(sub_problem)

    return problem_list


def code_verify(
    id2info,
    generateds,
    query_ids,
    timeout=FUNCTIONCALL_TIMEOUT,
    timeout_for_testcase=SINGLE_CASE_EXEC_TIMEOUT,
    test_case_batch_size=TEST_CASE_BATCH_SIZE,
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
        test_case_batch_size,
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
            logger.debug(
                f'Functioncall code verify not passed, uid: {rsp.get("uid")}, query id: {query_id}, results: {rsp}'
            )

        results[query_index] = results[query_index] and value

    logger.info(
        f"code_verify finished, request count: {len(payload_list)}, query count: {len(query_ids)}, result count: {len(results)}, code reward: {results}"
    )
    return results


if __name__ == "__main__":
    # data_list = load_jsonl("functioncall/test/test_success_dataset.jsonl")
    # id2info = defaultdict(dict)
    # for item in data_list:
    #     id2info[item["query_id"]] = item

    id2info = {
        "code_code_contests_26": {
            "task": "code",
            "query_id": "code_code_contests_26",
            "solutions": [
                'from sys import stdin, gettrace\n\nif gettrace():\n    def inputi():\n        return input()\nelse:\n    def input():\n        return next(stdin)[:-1]\n\n\n    def inputi():\n        return stdin.buffer.readline()\n\nLIMIT = 200001\nMOD = 1000000007\n\ndef solve(factorial):\n    n = int(input())\n    print((factorial[2 * n] * pow(2, MOD - 2, MOD)) % MOD)\n\ndef main():\n    factorial = [1]\n    for i in range(1, LIMIT):\n        factorial.append((factorial[-1]*i)%MOD)\n    t = int(input())\n    for _ in range(t):\n        solve(factorial)\n\n\nif __name__ == "__main__":\n    main()\n'
            ],
            "language": "PYTHON",
            "input_output": '{"inputs": ["4\\n1\\n2\\n9\\n91234\\n"], "outputs": ["1\\n12\\n830455698\\n890287984\\n"], "fn_name": "", "remote": false}',
            "prompt": [
                "<role>HUMAN</role>CQXYM is counting permutations length of 2n.\n\nA permutation is an array consisting of n distinct integers from 1 to n in arbitrary order. For example, [2,3,1,5,4] is a permutation, but [1,2,2] is not a permutation (2 appears twice in the array) and [1,3,4] is also not a permutation (n=3 but there is 4 in the array).\n\nA permutation p(length of 2n) will be counted only if the number of i satisfying p_i<p_{i+1} is no less than n. For example:\n\n  * Permutation [1, 2, 3, 4] will count, because the number of such i that p_i<p_{i+1} equals 3 (i = 1, i = 2, i = 3).\n  * Permutation [3, 2, 1, 4] won't count, because the number of such i that p_i<p_{i+1} equals 1 (i = 3). \n\n\n\nCQXYM wants you to help him to count the number of such permutations modulo 1000000007 (10^9+7).\n\nIn addition, [modulo operation](https://en.wikipedia.org/wiki/Modulo_operation) is to get the remainder. For example:\n\n  * 7 mod 3=1, because 7 = 3 ⋅ 2 + 1, \n  * 15 mod 4=3, because 15 = 4 ⋅ 3 + 3. \n\nInput\n\nThe input consists of multiple test cases. \n\nThe first line contains an integer t (t ≥ 1) — the number of test cases. The description of the test cases follows.\n\nOnly one line of each test case contains an integer n(1 ≤ n ≤ 10^5).\n\nIt is guaranteed that the sum of n over all test cases does not exceed 10^5\n\nOutput\n\nFor each test case, print the answer in a single line.\n\nExample\n\nInput\n\n\n4\n1\n2\n9\n91234\n\n\nOutput\n\n\n1\n12\n830455698\n890287984\n\nNote\n\nn=1, there is only one permutation that satisfies the condition: [1,2].\n\nIn permutation [1,2], p_1<p_2, and there is one i=1 satisfy the condition. Since 1 ≥ n, this permutation should be counted. In permutation [2,1], p_1>p_2. Because 0<n, this permutation should not be counted.\n\nn=2, there are 12 permutations: [1,2,3,4],[1,2,4,3],[1,3,2,4],[1,3,4,2],[1,4,2,3],[2,1,3,4],[2,3,1,4],[2,3,4,1],[2,4,1,3],[3,1,2,4],[3,4,1,2],[4,1,2,3].\n\nWrite Python code to solve the problem. Present the code in the code block:\n```python\nYour code\n```\n\n\nWrite Python code to solve the problem. Present the code in the code block:\n```python\nYour code\n```<role>ASSISTANT</role>"
            ],
        }
    }

    def create_test_params(count=10):
        query_ids = []
        generateds = []
        cnt = 0

        for _, d in id2info.items():
            if cnt >= count:
                break
            if d["query_id"] not in id2info:
                continue
            query_ids.append(d["query_id"])
            generateds.extend(d["solutions"])
            cnt += 1

        return generateds, query_ids

    generateds, query_ids = create_test_params(100)
    scale = 1
    print(f"generateds:, query_ids:{query_ids}")
    result = code_verify(id2info, generateds * scale, query_ids * scale)
    print(result)
