# Copyright 2025 Ant Group Inc.

import json
import os
import signal
import subprocess
import uuid
from typing import *

from realhf.base import logging

logger = logging.getLogger("math parser")


def get_box(s):
    pos = -1
    cnt = 0
    for i in range(len(s)):
        if s[i] == "{":
            cnt += 1
            if cnt == 1:
                pos = i + 1
        if s[i] == "}":
            cnt -= 1
            if cnt == 0:
                return s[pos:i]

    return None


def get_answer(answer):
    pos = answer.find("\\boxed{")
    if pos == -1:
        return []
    return [get_box(answer[pos:])] + get_answer(answer[pos + 1 :])


def loadJson(dataDir):
    with open(dataDir, "r") as f:
        if dataDir.endswith(".jsonl"):
            samples = [json.loads(line) for line in f.readlines()]
        else:
            samples = json.load(f)

    return samples


def parse_line(id2info, prompt_str, generated, query_id):
    info = id2info[query_id.split("@idx:")[0]]

    tmp_id = str(uuid.uuid4())
    with open(f"/tmp/{tmp_id}-input.jsonl", "w", encoding="utf-8") as f:
        for cur_solution in info["solutions"]:
            f.write(json.dumps({"answer": generated, "solution": cur_solution}) + "\n")

    venv_python = "/sympy/bin/python3"
    # logger.info(f"math verify working dir: `{os.getcwd()}`")
    pro = subprocess.Popen(
        " ".join(
            [
                venv_python,
                "math_verify_utils_qwen.py",
                "--tmp_id",
                tmp_id,
            ]
        ),
        shell=True,
        preexec_fn=os.setsid,
        stdout=subprocess.DEVNULL,
    )
    pro.wait()
    try:
        os.killpg(os.getpgid(pro.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass

    label = 0
    try:
        with open(f"/tmp/{tmp_id}-output.jsonl", "r") as f:
            for line in f.readlines():
                output_data = json.loads(line)
                label = output_data["retval"] or label
    except FileNotFoundError as e:
        # The subprocess may fail to parse the input (maybe due to reaching the maximum recursion length)
        # We just return 0 for the reward.
        logger.warning(
            f"Failed to parse: query_id `{query_id}`, prompt `{prompt_str}`, seq `{generated}`. Set 0 reward."
        )
        label = 0
    finally:
        if os.path.exists(f"/tmp/{tmp_id}-input.jsonl"):
            os.remove(f"/tmp/{tmp_id}-input.jsonl")
        if os.path.exists(f"/tmp/{tmp_id}-output.jsonl"):
            os.remove(f"/tmp/{tmp_id}-output.jsonl")
        return label


def parse_lines_in_parallel(
    id2info,
    generateds: List,
    query_ids: List,
    max_workers=22,
    check_xml_format=False,
) -> List:
    assert len(generateds) == len(query_ids), (
        len(generateds),
        len(query_ids),
    )
    bs = len(query_ids)
    mbs = (bs + max_workers - 1) // max_workers

    tmp_ids = []
    all_query_indices = []
    for i in range(max_workers):
        tmp_id = str(uuid.uuid4())
        query_indices = []
        s = slice(i * mbs, (i + 1) * mbs)
        offset = i * mbs
        with open(f"/tmp/{tmp_id}-input.jsonl", "w", encoding="utf-8") as f:
            for idx, (query_id, generated) in enumerate(
                zip(query_ids[s], generateds[s])
            ):
                info = id2info[query_id.split("@idx:")[0]]
                for cur_solution in info["solutions"]:
                    f.write(
                        json.dumps({"answer": generated, "solution": cur_solution})
                        + "\n"
                    )
                    query_indices.append(idx + offset)
        tmp_ids.append(tmp_id)
        all_query_indices.append(query_indices)

    venv_python = "/sympy/bin/python3"
    # logger.info(f"math verify working dir: `{os.getcwd()}`")
    procs = []
    for tmp_id in tmp_ids:
        pro = subprocess.Popen(
            " ".join(
                [
                    venv_python,
                    "math_verify_utils_qwen.py",
                    "--tmp_id",
                    tmp_id,
                    # "--check_xml_format",
                    # "True" if check_xml_format else "False",
                ]
            ),
            shell=True,
            preexec_fn=os.setsid,
            stdout=subprocess.DEVNULL,
        )
        procs.append(pro)
    for pro in procs:
        try:
            pro.wait()
        except Exception as e:
            pass
        try:
            os.killpg(os.getpgid(pro.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass

    labels = [0 for _ in query_ids]
    for i, (tmp_id, query_indices) in enumerate(zip(tmp_ids, all_query_indices)):
        try:
            with open(f"/tmp/{tmp_id}-output.jsonl", "r") as f:
                for _ansidx, line in enumerate(f.readlines()):
                    output_data = json.loads(line)
                    labels[query_indices[_ansidx]] = (
                        output_data["retval"] or labels[query_indices[_ansidx]]
                    )
        except FileNotFoundError as e:
            # The subprocess may fail to parse the input (maybe due to reaching the maximum recursion length)
            # We just return 0 for the reward.
            logger.warning(f"Failed to parse generated answers. Set 0 reward.")
        finally:
            if os.path.exists(f"/tmp/{tmp_id}-input.jsonl"):
                os.remove(f"/tmp/{tmp_id}-input.jsonl")
            if os.path.exists(f"/tmp/{tmp_id}-output.jsonl"):
                os.remove(f"/tmp/{tmp_id}-output.jsonl")
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

    print(
        parse_lines_in_parallel(
            id2info,
            sample["answers"] * 100,
            ["fe11b471-1aa9-4867-958f-a0a811c85f92" for _ in range(100)],
            max_workers=8,
            check_xml_format=True,
        )
    )
