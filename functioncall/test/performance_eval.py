import math
import os
import json
import time
from collections import defaultdict
import random
from multiprocessing import Pool, Manager, cpu_count, shared_memory
from typing import List, Dict, Any
import pickle
import numpy as np
from realhf.base import logging
from functioncall.code.verify import code_verify

logger = logging.getLogger("function call")

def parallel_code_verify(
    id2info: Dict[str, Any],
    generateds: List[str],
    query_ids: List[str],
    verbose: bool = True,
    num_processes: int = min(cpu_count(), 128),
) -> List[Any]:

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
        )
        for i in range(0, len(generateds), chunk_size)
    ]

    print(
        f"parallel_code_verify start generateds_size: {len(generateds)}, query_ids_size:{len(query_ids)}, {num_processes} processes"
        f"using "
    )

    with Pool(processes=num_processes) as pool:
        start_time = time.time()
        chunk_results = pool.starmap(process_ordered_chunk, chunks)
        flat_results = [item for chunk in chunk_results for item in chunk]

        duration = time.time() - start_time
        print(
            f"Processed {len(generateds)} items in {duration:.2f} seconds "
            f"using {num_processes} processes"
        )

    shm.close()
    shm.unlink()
    return flat_results


def process_ordered_chunk(
    index,
    shared_dict,
    generateds,
    query_ids,
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

        results = code_verify(id2info, generateds, query_ids, True)
        if len(results) != len(generateds):
            raise ValueError(
                f"Result length mismatch: expected {len(generateds)}, got {len(results)}"
            )
        logger.info(f"Process {index} completed in {time.monotonic()-start:.2f}s")
        return results
    except pickle.UnpicklingError as e:
        logger.error(f"Failed to deserialize shared memory: {e}")
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


if __name__ == "__main__":
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
    print(f"Total results: {result}")
    logger.info(
        f"Process results: {result}, size: {len(generateds)}, in {time.time()-start_time:.2f}s"
    )
