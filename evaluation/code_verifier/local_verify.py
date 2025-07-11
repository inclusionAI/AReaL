import concurrent.futures
import json
import os
import signal
import subprocess
import sys
import time
import traceback
import uuid
from collections import defaultdict
from io import StringIO

import numpy as np

SINGLE_CASE_EXEC_TIMEOUT = 6

import logging

from utils import load_jsonl

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


def call_verify(problem, generation, debug, timeout=SINGLE_CASE_EXEC_TIMEOUT):

    tmp_id = str(uuid.uuid4())
    input_data = {
        "sample": problem,
        "test": generation,
        "debug": debug,
        "timeout": timeout,
    }
    with open(f"/tmp/{tmp_id}-input.json", "w") as temp_file:
        json.dump(input_data, temp_file)
    start_time = time.time()

    venv_python = "python3"
    pro = subprocess.Popen(
        " ".join(
            [
                venv_python,
                "code_verifier/testing_util.py",
                "--tmp_id",
                tmp_id,
            ]
        ),
        shell=True,
        preexec_fn=os.setsid,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        pro.wait(600)
    except Exception as e:
        pass
    try:
        os.killpg(os.getpgid(pro.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass

    result = {"result": [False], "info": {}}
    try:
        with open(f"/tmp/{tmp_id}-output.json", "r") as f:
            result = json.load(f)
    except FileNotFoundError as e:
        logger.warning(
            f"{problem['query_id']}: Failed to parse generated answers. FileNotFoundError. Set 0 reward."
        )
    except Exception as e:
        logger.warning(
            f"{problem['query_id']}: Failed to parse generated answers. {e}. Set 0 reward."
        )
    finally:
        if os.path.exists(f"/tmp/{tmp_id}-input.json"):
            os.remove(f"/tmp/{tmp_id}-input.json")
        if os.path.exists(f"/tmp/{tmp_id}-output.json"):
            os.remove(f"/tmp/{tmp_id}-output.json")

    execution_time = time.time() - start_time
    logger.info(
        f'[call_verify] query_id: {problem["query_id"]}, start_time: {str(start_time)}, Time elapsed: {execution_time * 1000:.0f} ms'
    )
    return result["result"], result["info"]


def code_verify(id2info, generateds, query_ids, debug=False):
    assert len(generateds) == len(query_ids)
    problems = [id2info[qid] for qid in query_ids]

    final_results = []

    infer_args = []
    for query_id, generated, problem in zip(query_ids, generateds, problems):
        infer_args.append((problem, generated, debug, SINGLE_CASE_EXEC_TIMEOUT))

    run_results = []
    num_process = os.cpu_count()
    with concurrent.futures.ProcessPoolExecutor(num_process) as executor:
        run_results = executor.map(call_verify, *zip(*infer_args))

    for run_result in run_results:
        curr_res, metadata = run_result
        if any(x != True for x in curr_res):
            final_results.append(0)
        else:
            final_results.append(1)

    return final_results


def evaluate(samples):

    infer_args = []
    scores = []
    for sample in samples:
        for pred in sample["pred"]:
            problem = {
                "input_output": sample["input_output"],
                "query_id": sample["idx"],
            }
            infer_args.append((problem, pred, False, SINGLE_CASE_EXEC_TIMEOUT))

    run_results = []
    num_process = os.cpu_count()
    with concurrent.futures.ProcessPoolExecutor(num_process) as executor:
        run_results = executor.map(call_verify, *zip(*infer_args))

    for run_result in run_results:
        curr_res, metadata = run_result
        if any(x != True for x in curr_res):
            scores.append(0)
        else:
            scores.append(1)

    idx = 0
    score_mat = []
    for sample in samples:
        sample["score"] = scores[idx : idx + len(sample["pred"])]
        assert len(sample["score"]) == len(sample["pred"])
        score_mat.append(sample["score"])
        idx += len(sample["pred"])

    col_means = np.array(score_mat).mean(axis=0)
    mean_score = list(np.round(col_means * 100, decimals=1))

    max_len = max([len(s) for s in score_mat])

    for i, s in enumerate(score_mat):
        assert len(s) == max_len

    result_json = {
        "num_samples": len(samples),
        "num_scores": len(scores),
        "empty_samples": len([s for s in samples if not s["pred"][-1]]),
        "acc": np.mean(mean_score),
    }

    return samples, result_json


if __name__ == "__main__":
    # data_list = load_jsonl("/storage/openpsi/data/functioncall/test/test_success_dataset.jsonl")
    data_path = "/storage/openpsi/data/code/deepcoder/deepcoder_0415_v3_verify_new_correct.jsonl"

    id2info = defaultdict(dict)
    for item in load_jsonl(data_path):
        id2info[item["query_id"]] = item

    def create_test_params(count=10):
        query_ids = []
        generateds = []
        cnt = 0

        for d in load_jsonl(data_path):
            if cnt >= count:
                break
            if not d["solutions"] or d["query_id"] not in id2info:
                continue
            query_ids.append(d["query_id"])
            generateds.append(d["solutions"][0])
            cnt += 1

        return generateds, query_ids

    generateds, query_ids = create_test_params(100)
    scale = 1
    print(f"generateds:, query_ids:{query_ids}")
    result = code_verify(id2info, generateds * scale, query_ids * scale)
    print(result)
