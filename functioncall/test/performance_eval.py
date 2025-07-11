import copy
import json
import logging
import math
import os
import pickle
import sys
import time
from collections import defaultdict
from datetime import datetime
from multiprocessing import Manager, Pool, cpu_count, shared_memory
from typing import Any, Dict, List

import numpy as np

from functioncall.code.verify import code_verify
from realhf.utils import load_hf_or_local_file

logger = logging.getLogger("function call")


def parallel_code_verify(
    id2info: Dict[str, Any],
    generateds: List[str],
    query_ids: List[str],
    test_case_batch_size: int,
    num_processes: int = min(cpu_count(), 128),
) -> List[Any]:
    shm = None
    pool = None
    try:
        # set id2info in shared memory
        serialized_dict = pickle.dumps(id2info)
        buffer = np.frombuffer(serialized_dict, dtype=np.uint8)
        shm = shared_memory.SharedMemory(create=True, size=buffer.nbytes)
        buffer_shared = np.ndarray(buffer.shape, dtype=buffer.dtype, buffer=shm.buf)
        buffer_shared[:] = buffer[:]
        shared_dict = (shm.name, buffer.shape, buffer.dtype)

        chunk_size = math.ceil(len(generateds) / num_processes)
        chunks = [
            (
                i,
                shared_dict,
                generateds[i : i + chunk_size],
                query_ids[i : i + chunk_size],
                test_case_batch_size,
            )
            for i in range(0, len(generateds), chunk_size)
        ]

        print(
            f"parallel_code_verify start generateds_size: {len(generateds)}, query_ids_size:{len(query_ids)}, {num_processes} processes"
            f"using "
        )

        pool = Pool(processes=num_processes)
        start_time = time.time()
        chunk_results = pool.starmap(process_ordered_chunk, chunks)
        flat_results = [item for chunk in chunk_results for item in chunk]

        duration = time.time() - start_time
        print(
            f"Processed {len(generateds)} items in {duration:.2f} seconds "
            f"using {num_processes} processes"
        )

        return flat_results

    except KeyboardInterrupt:
        print("\nReceived Ctrl+C, terminating processes...")
        if pool is not None:
            pool.terminate()
            pool.join()
        return []
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return []
    finally:
        if shm is not None:
            shm.close()
            shm.unlink()
        if pool is not None:
            pool.close()


def process_ordered_chunk(
    index,
    shared_dict,
    generateds,
    query_ids,
    test_case_batch_size,
) -> List[tuple[int, Any]]:
    start = time.monotonic()
    logger.info(
        f"Process start at {start}s, chunk_index: {index}, chunk_size: {len(generateds)}, query_size: {len(query_ids)}"
    )

    try:
        shm_name, shape, dtype = shared_dict
        existing_shm = shared_memory.SharedMemory(name=shm_name)
        buffer = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
        id2info = pickle.loads(buffer.tobytes())
        results = code_verify(
            id2info, generateds, query_ids, test_case_batch_size=test_case_batch_size
        )
        if len(results) != len(generateds):
            raise ValueError(
                f"Result length mismatch: expected {len(generateds)}, got {len(results)}"
            )
        logger.info(f"Process {index} completed in {time.monotonic()-start:.2f}s")
        return results
    except pickle.UnpicklingError as e:
        logger.error(f"Failed to deserialize shared memory: {e}")
        return [str(e)] * len(query_ids)
    except Exception as e:
        logger.error(
            f"Process {index} failed in {time.monotonic() - start:.2f}s, err: {str(e)}"
        )
        return [str(e)] * len(query_ids)
    finally:
        if "existing_shm" in locals():
            existing_shm.close()


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


def save_jsonl(samples, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print("Saved to", save_path)


def load_jsonl_stream(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def lcb_dataset_eval():
    data4 = load_jsonl(
        "/storage/openpsi/data/code/live_code_bench/live_code_bench_v4_v5-r1-distilled-prompt-fnname.jsonl"
    )
    id2info = defaultdict(dict)
    for item in data4:
        query_id = str(item["query_id"])
        id2info[query_id] = item

    def create_test_params(count=-1):
        query_ids = []
        generateds = []
        cnt = 0

        file_path = "/storage/openpsi/users/meijun.mei/datasets/Scenario.codegeneration_10_0.2_eval_all.json"
        raw_data = []
        with open(file_path, "r", encoding="utf-8") as f:
            raw_data = [line for line in json.load(f)]

        for d in raw_data:
            if count > 0 and cnt >= count:
                break
            if not d["code_list"] or d["question_id"] not in id2info:
                continue

            generateds.extend(d["code_list"])
            query_ids.extend([d["question_id"]] * len(d["code_list"]))
            cnt += len(d["code_list"])

        return generateds, query_ids

    generateds, query_ids = create_test_params()
    start_time = time.time()
    scale = 2
    result = parallel_code_verify(
        id2info, generateds * scale, query_ids * scale, num_processes=16
    )

    # vals, metas =
    print(f"Total results: {result}")
    logger.info(
        f"Process results: {result}, size: {len(generateds)}, in {time.time()-start_time:.2f}s"
    )


def build_sol_id(query_id, solution_index):
    return query_id + f"[solution{solution_index}]"


def parse_sol_id(sol_id):
    query_id, solution_part = sol_id.split("[", 1)
    solution_content = solution_part.rstrip("]")
    return query_id, solution_content


def statics_result(result, query_ids):
    result_statistics = defaultdict(
        lambda: {"query_id": "", "pass": True, "solutions": []}
    )
    for i, query_id in enumerate(query_ids):
        org_id, sol_idx = parse_sol_id(query_id)
        result_statistics[org_id]["query_id"] = org_id
        result_statistics[org_id]["solutions"].append({sol_idx: bool(result[i])})
        result_statistics[org_id]["solutions"] = sorted(
            result_statistics[org_id]["solutions"], key=lambda x: list(x.keys())[0]
        )
        if not result[i]:
            result_statistics[org_id]["pass"] = False

    return list(result_statistics.values())


def standard_dataset_eval(
    dataset_path, code_count=0, test_case_batch_size=20, dry_run=False
):
    dataset_path = load_hf_or_local_file(dataset_path)
    id2info = defaultdict(dict)
    generateds, query_ids = [], []
    cnt = 0
    testcase_in_dataset = 0
    testcases_in_runtime = 0
    request_size = 0

    for item in load_jsonl_stream(dataset_path):
        if code_count and cnt >= code_count:
            break
        if not item["solutions"]:
            continue
        generateds.extend(item["solutions"])

        # set unique query_id for each solution code
        for i in range(len(item["solutions"])):
            query_id = build_sol_id(item["query_id"], i)
            query_ids.append(query_id)
            id2info[query_id] = copy.copy(item)
            id2info[query_id]["query_id"] = query_id

        # metrics
        case_size = sys.getsizeof(item["input_output"])
        assert (
            case_size < 500 * 1024
        ), f"'input_output' exceeds 500KB ({case_size} bytes). Use remote testcase instead."
        cnt += len(item["solutions"])
        case_size = len(json.loads(item["input_output"]).get("inputs", []))
        testcase_in_dataset += case_size
        testcases_in_runtime += case_size * len(item["solutions"])
        request_size += math.ceil(case_size / test_case_batch_size) * len(
            item["solutions"]
        )

    start_time = time.time()
    logger.info(
        f"Start process, code size: {len(generateds)}, request size: {request_size}, testcase_in_dataset: {testcase_in_dataset}, testcases_in_runtime: {testcases_in_runtime}"
    )
    if dry_run:
        return
    result = parallel_code_verify(
        id2info, generateds, query_ids, test_case_batch_size, num_processes=16
    )

    # passed solutions
    solution_pass_rate = result.count(1) / len(result)

    logger.info(
        f"Process results: {result}, code size: {len(generateds)},request size: {request_size}, testcase_in_dataset: {testcase_in_dataset}, testcases_in_runtime: {testcases_in_runtime}, solution_pass_rate:{solution_pass_rate} in {time.time()-start_time:.2f}s"
    )

    result_statistics = statics_result(result, query_ids)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_jsonl(
        result_statistics,
        os.path.basename(dataset_path) + f"{timestamp}.stat",
    )


if __name__ == "__main__":
    # lcb_dataset_eval()

    standard_dataset_eval(
        "/storage/openpsi/users/meijun.mei/datasets/loj_0410_format2.jsonl",
        code_count=0,
        dry_run=False,
    )

    # standard_dataset_eval(
    #     "/storage/openpsi/data/code/live_code_bench_for_test/live_code_bench_v4_v5-for-test-remote.jsonl",
    #     code_count=0,
    #     dry_run=False,
    # )
