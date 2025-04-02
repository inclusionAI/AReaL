import json
import multiprocessing
import os
import sys
import time
import traceback
from io import StringIO
from typing import Dict, List

from functioncall.code.function.testing_util import run_test
from realhf.base import logging

logger = logging.getLogger("function call")


def capture_stdout(code):
    original_stdout = sys.stdout
    fake_stdout = StringIO()

    try:
        sys.stdout = fake_stdout
        exec(code, {"__builtins__": __builtins__})
    except Exception as e:
        return f"error: {str(e)}, traceback: {traceback.format_exc()}"
    finally:
        sys.stdout = original_stdout
    return fake_stdout.getvalue()


def _temp_run(problem, generation, debug, result):
    start_time = time.time()

    try:
        if debug:
            logger.debug(f"Running test for problem: {problem}")
        r = run_test(sample=problem, test=generation, debug=debug)
        result.append(r)
        if debug:
            logger.debug(f"Test completed with result: {result}")
    except Exception as e:

        logger.warning(
            f"Error in _temp_run: {e}\n"
            f"traceback: {''.join(traceback.format_exception(*sys.exc_info()))}\n"
            f"problem:{problem}"
        )

    execution_time = time.time() - start_time
    logger.info(
        f'[_temp_run] query_id: {problem["problem_id"]}, start_time: {str(start_time)}, Time elapsed: {execution_time * 1000:.0f} ms'
    )


def check_correctness(problem, generation, timeout, debug=False):
    """Check correctness of code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`"""
    if debug:
        # FIXME: error variable "problem" is not defined
        result = capture_stdout(
            "from functioncall.code.function.testing_util import run_test\n"
            + "run_test(sample=problem, test=generation, debug=debug)"
        )
        return result[0], result[1]

    start_time = time.time()
    manager = multiprocessing.Manager()
    result = manager.list()
    p = multiprocessing.Process(
        target=_temp_run, args=(problem, generation, debug, result)
    )
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        if debug:
            logger.debug(f"Process is still alive. Killing the process.")
        p.kill()
    if not result:
        # Remark: ideally we would consider that all tests failed but we can't access number of tests here easily
        # so we use 21=the average number of tests for a smaple in the test split instead
        avg_number_tests = 21
        result = [[-1 for _ in range(avg_number_tests)], {}]
        if debug:
            logger.debug(f"Global timeout occurred, returning default result.")
    if debug:
        logger.debug(f"Final result: {result}")

    execution_time = time.time() - start_time
    logger.info(
        f'[check_correctness] query_id: {problem["problem_id"]}, start_time: {str(start_time)}, Time elapsed: {execution_time * 1000:.0f} ms'
    )
    return result[0]


def code_verify(id2info, generateds, query_ids, debug=False):
    assert len(generateds) == len(query_ids)
    problems = [id2info[qid] for qid in query_ids]

    result = []

    for query_id, generated, problem in zip(query_ids, generateds, problems):
        logger.debug(f"run_batch_code, query_id: {query_id}")
        try:
            curr_res, metadata = check_correctness(
                problem=problem, generation=generated, timeout=6000, debug=debug
            )

            if any(x != True for x in curr_res):
                logger.debug(f"id:{query_id}, Results were not all True: {metadata}")
                result.append(0)
            else:
                result.append(1)

        except Exception as e:
            exc_info = sys.exc_info()
            logger.error(
                f"test framework exception = {repr(e)}{e}\n{traceback.format_exception(*exc_info)}"
            )
            result.append(0)

    return result


if __name__ == "__main__":
    path = "/storage/openpsi/data/code/apps/test.jsonl"
    data = []
    with open(path, "r") as f:
        code_data = [json.loads(l) for l in f.readlines()]

    id2info = {}
    solutions = []
    query_ids = []
    for i in range(10):
        problem = code_data[i]
        problem["problem_id"] = problem["id"]
        id2info[problem["problem_id"]] = problem
        solutions.append(json.loads(problem["solutions"])[0])
        query_ids.append(problem["id"])

    result = code_verify(
        id2info,
        solutions,
        query_ids,
        debug=False,
    )
    print(result)
