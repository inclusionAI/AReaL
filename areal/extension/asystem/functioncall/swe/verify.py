import asyncio
import json
import os
from collections import defaultdict

from areal.extension.asystem.functioncall.base.call import (
    Language,
    batch_function_call,
    get_runtime_name,
)
from areal.extension.asystem.functioncall.base.utils import (
    construct_swe_uid,
    load_jsonl,
    logger,
)
from areal.extension.asystem.functioncall.swe.utils import (
    convert_swe_output_to_patch,
)

TEST_CASE_BATCH_SIZE = 1
FUNCTIONCALL_TIMEOUT = 600

swe_config_dict = json.loads(os.getenv("REWARD_FUNCTIONCALL_CONFIG", "{}")).get(
    "swe", {}
)


def load_problems_with_testcase_batch(id2info, query_ids, generateds):
    problem_list = []

    for _, (query_id, generated) in enumerate(zip(query_ids, generateds)):
        problem = id2info[query_id]
        query_id = problem.get("query_id", "")
        extra_info = problem.get("extra_info")
        runtime = query_id
        test_patch = extra_info.get("test_patch", "")
        commitId = extra_info.get("base_commit", "")
        evalTestcases = extra_info.get("evalTestcases", [])
        code_context = json.loads(extra_info["code_context"])
        uid = construct_swe_uid(query_id)

        generated_patch = convert_swe_output_to_patch(uid, code_context, generated)

        problem = {
            "type": "swe",
            "runtime": runtime,
            "language": "python",
            "workspace": "/testbed",
            "commitId": commitId,
            "timeout": 300,
            "extraInfo": {
                "queryId": uid,
                "patches": [test_patch, generated_patch],
                "evalTestcases": evalTestcases,
            },
        }

        problem_list.append(problem)
    return problem_list


async def swe_verify(
    id2info,
    generateds,
    query_ids,
    timeout=FUNCTIONCALL_TIMEOUT,
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
    )

    logger.info(
        f"swe_verify start, request count: {len(payload_list)}, payload_list: {payload_list}, query_id_0: {query_ids[0]}"
    )

    swe_host = swe_config_dict.get("host", "")
    rsp_list = await batch_function_call(payload_list, "swe", timeout, host=swe_host)

    results = [1] * len(query_ids) if len(rsp_list) else [0] * len(query_ids)
    for idx, rsp in enumerate(rsp_list):
        query_id = query_ids[idx]

        value = 0
        if rsp and rsp.get("success", False):
            value = 1
        else:
            logger.info(
                f'Functioncall swe verify not passed, uid: {rsp.get("uid")}, query id: {query_id}, results: {rsp}'
            )

        results[idx] = value

    logger.info(
        f"swe_verify finished, request count: {len(payload_list)}, query count: {len(query_ids)}, result count: {len(results)}, swe reward: {results}"
    )
    return results


if __name__ == "__main__":
    data_list = load_jsonl("areal/extension/asystem/functioncall/swe/swe-dataset.jsonl")
    id2info = defaultdict(dict)
    for item in data_list:
        id2info[item["query_id"]] = item

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
            generateds.append(d["solutions"][0])
            cnt += 1

        return generateds, query_ids

    generateds, query_ids = create_test_params(1)
    scale = 1
    print(f"generateds:, query_ids:{query_ids}")
    result = asyncio.run(swe_verify(id2info, generateds * scale, query_ids * scale))
    print(result)
