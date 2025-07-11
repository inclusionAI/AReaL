# Copyright 2025 Ant Group Inc.

import ast
import asyncio
import dataclasses
import html
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple

import colorama
import numpy as np
import torch
import torch.distributed as dist

import realhf.api.core.model_api as model_api
import realhf.base.logging as logging
from functioncall.code.local_verify import code_verify as local_code_verify
from functioncall.code.verify import code_verify
from functioncall.math.verify import math_verify
from realhf.api.core.data_api import (
    RL_TASKS,
    MicroBatchSpec,
    SequenceSample,
    load_hf_tokenizer,
)
from realhf.base import constants
from realhf.base.datapack import flat2d
from realhf.impl.dataset.math_code_dataset import load_metadata
from realhf.impl.dataset.math_parser import parse_lines_in_parallel as math_verify_local

logger = logging.getLogger("Packed Reward Modeling Interface", "benchmark")

ENABLE_FUNCTION_CALL = True if os.getenv("FUNCTIONCALL_SERVICE_DOMAIN", "") else False
math_verify_call = math_verify if ENABLE_FUNCTION_CALL else math_verify_local
code_verify_call = code_verify if ENABLE_FUNCTION_CALL else local_code_verify


class VerifierException(Exception):
    pass


def extract_python_code(text, min_length=20, strict_syntax=True):
    code_pattern = r"(?i)```(?:python|py)?\s*\n?(.*?)\n?```"
    code_blocks = re.findall(code_pattern, text, re.DOTALL)
    valid_blocks = []
    for block in code_blocks:
        clean_block = block.strip()
        if len(clean_block) < min_length:
            continue

        # verify code syntax
        if strict_syntax:
            try:
                ast.parse(clean_block, mode="exec")
            except (SyntaxError, IndentationError):
                continue

        valid_blocks.append(clean_block)

    if not valid_blocks:
        # logger.warning(f"failed to extract python code from {text}")
        return None
    # return the last code block
    return valid_blocks[-1]


def check_with_elementtree(text):
    def escape_between_tags(text, tags=["think", "answer"]):
        """转义标签之间的内容，但保留标签本身."""
        # 构建标签模式
        tag_pattern = "|".join(tags)
        parts = []
        current_pos = 0

        # 匹配开始和结束标签
        pattern = f"</?({tag_pattern})[^>]*>"

        for match in re.finditer(pattern, text):
            # 添加标签之前的内容（需要转义）
            if current_pos < match.start():
                parts.append(html.escape(text[current_pos : match.start()]))

            # 添加标签本身（不转义）
            parts.append(match.group())
            current_pos = match.end()

        # 添加最后剩余的内容
        if current_pos < len(text):
            parts.append(html.escape(text[current_pos:]))

        return "".join(parts)

    text = escape_between_tags(text)
    if not text.strip().startswith("<think>"):
        text = "<think>" + text
    try:
        xml_text = f"<root>{text}</root>"
        x = ET.fromstring(xml_text)
        if x.text is not None and x.text.strip() != "":
            return False, f"Error: extra content before <think>. {x.text}"
        if len(x) != 2:
            return False, f"Error: there are {len(x)} tags."
        if x[0].tag is None or x[0].tag != "think":
            return False, f"Error: <think> tag is missing. {x[0].tag}"
        if x[0].tail is not None and x[0].tail.strip() != "":
            return (
                False,
                f"Error: extra content between <think> and <answer>. {x[0].tail}",
            )
        if x[1].tag is None or x[1].tag != "answer":
            return False, f"Error: <answer> tag is missing. {x[1].tag}"
        if x[1].tail is not None and x[1].tail.strip() != "":
            return False, f"Error: extra content after <answer>, {x[1].tail}"

        return True, x[1].text if x[1].text is not None else ""
    except ET.ParseError as e:
        return False, f"Error: XML格式错误, {str(e)}"


id2info = {}


def dispatch_reward_calculation(task, answers, query_id_strs) -> List:
    global id2info
    assert len(answers) == len(query_id_strs)
    format_rewards = []
    if task == "math" or task == "stem":
        format_rewards = math_verify_call(id2info, answers, query_id_strs)
    elif task == "code":
        codes = [extract_python_code(_answer) for _answer in answers]
        format_rewards = code_verify_call(id2info, codes, query_id_strs)
    assert len(format_rewards) == len(answers), (
        task,
        len(format_rewards),
        len(answers),
        answers,
    )
    return format_rewards


def retokenize_and_verify(
    task,
    tokenizer,
    prompt_ids: List[List[int]],
    seq_ids: List[List[int]],
    query_ids: List[str],
    check_xml_format=False,
):
    seq_strs = tokenizer.batch_decode(
        seq_ids, clean_up_tokenization_spaces=False, skip_special_tokens=True
    )
    prompt_strs = tokenizer.batch_decode(
        prompt_ids, clean_up_tokenization_spaces=False, skip_special_tokens=True
    )
    # query_id_strs = query_ids
    query_id_strs = [query_id.split("@")[0] for query_id in query_ids]

    answers = [
        seq_str.split(prompt_str)[1]
        for seq_str, prompt_str in zip(seq_strs, prompt_strs)
    ]

    format_rewards = dispatch_reward_calculation(task, answers, query_id_strs)

    if check_xml_format:
        for idx, answer in enumerate(answers):
            xml_reward, _ = check_with_elementtree(answer)
            if xml_reward == 1 and format_rewards[idx] == 0:
                format_rewards[idx] = -0.8
            elif xml_reward == 0 and format_rewards[idx] == 0:
                format_rewards[idx] = -1

    return format_rewards, prompt_strs, seq_strs


@dataclasses.dataclass
class MultiTaskRewardInterface(model_api.ModelInterface):
    dataset_path: str = ""
    tokenizer_path: str = "/storage/models/Qwen__Qwen2.5-1.5B"
    answer_save_path: str = "."
    output_scaling: float = 1.0
    output_bias: float = 0.0
    rw_type: str = "sparse"
    check_xml_format: bool = False
    group_size: int = 1
    check_verifier_status: bool = False

    def __post_init__(self):
        global id2info
        id2info, _ = load_metadata(self.dataset_path)
        self.tokenizer = load_hf_tokenizer(self.tokenizer_path)
        if constants.parallelism_rank() == 0:
            logger.info(f"output_scaling: {self.output_scaling}")
            logger.info(f"output_bias: {self.output_bias}")
            logger.info(f"rw_type: {self.rw_type}")

    def _dispatch_tasks(self, data: SequenceSample) -> Tuple[Dict, Dict]:
        xs = data.unpack()
        dispatched = {}
        dispatched_indices = {}
        for task_idx, task_name in enumerate(RL_TASKS):
            indices = (
                (data.data["task_ids"] == task_idx).cpu().numpy().nonzero()[0].tolist()
            )
            if len(indices) > 0:
                dispatched[task_name] = SequenceSample.gather([xs[i] for i in indices])
                dispatched_indices[task_name] = indices

        return dispatched, dispatched_indices

    def _gather_tasks(
        self, results: Dict, dispatched_indices: Dict, bs: int
    ) -> SequenceSample:
        xs = [None for _ in range(bs)]
        for task_name, indices in dispatched_indices.items():
            xxs = results[task_name].unpack()
            assert len(indices) == len(xxs), (len(indices), len(xxs))
            for i, xx in zip(indices, xxs):
                xs[i] = xx
        assert all(xs)
        return SequenceSample.gather(xs)

    def _dispatch_tp_and_pp(self, data: SequenceSample):
        tp_pp_size = constants.tp_and_pp_world_size()
        if tp_pp_size == 1:
            return data, None
        splitted, _, backward_indices = data.split(
            mb_spec=MicroBatchSpec(n_mbs=tp_pp_size)
        )
        tp_pp_rank = constants.tp_and_pp_rank()
        print("dispatched batch size", [s.bs for s in splitted], flush=True)
        return splitted[tp_pp_rank], backward_indices

    def _gather_tp_and_pp(self, input_, data: SequenceSample, backward_indices):
        tp_pp_size = constants.tp_and_pp_world_size()
        if tp_pp_size == 1:
            return data
        local_rank = constants.grid().topo.get_rank(
            data=constants.data_parallel_rank(),
            tensor=0,
            pipe=constants.pipe_parallel_world_size() - 1,
        )
        dst = constants.to_global_pg_rank(local_rank)
        gather_list = None
        if dist.get_rank() == dst:
            gather_list = [None for _ in range(tp_pp_size)]
        x = data.data["rewards"].cpu().numpy().tolist()
        print(x, flush=True)
        dist.gather_object(
            x, gather_list, dst=dst, group=constants.tp_and_pp_cpu_group()
        )
        if dist.get_rank() != dst:
            return None
        gathered = np.array(gather_list).reshape(-1, self.group_size)
        assert len(gathered) == len(backward_indices)
        rewards = (
            np.concatenate([gathered[i] for i in backward_indices]).flatten().tolist()
        )
        return SequenceSample(
            keys=["rewards"],
            trailing_shapes=dict(rewards=()),
            dtypes=dict(rewards=torch.float32),
            ids=input_.ids,
            seqlens=dict(
                rewards=[[1 for _ in range(self.group_size)] for _ in range(input_.bs)],
            ),
            data=dict(rewards=torch.tensor(rewards, dtype=torch.float32)),
        )

    def calculate_task_reward(
        self,
        model: model_api.Model,
        data: SequenceSample,
        mb_spec: MicroBatchSpec,
        task_type: str,
    ):
        # mb_spec is disrespected here
        packed_input_ids: torch.Tensor = data.data["packed_input_ids"]
        input_seqlens = flat2d(data.seqlens["packed_input_ids"])
        seq_ids = []
        offset = 0
        for slen in input_seqlens:
            seq_ids.append(
                packed_input_ids[offset : offset + slen].cpu().numpy().tolist()
            )
            offset += slen
        assert offset == packed_input_ids.shape[0], (offset, packed_input_ids.shape)
        prompt_input_ids = data.data["packed_prompts"]
        prompt_len = flat2d(data.seqlens["packed_prompts"])
        prompt_ids = []
        offset = 0
        for slen in prompt_len:
            p = prompt_input_ids[offset : offset + slen].cpu().numpy().tolist()
            prompt_ids += [p] * self.group_size
            offset += slen
        format_rewards, prompt_strs, seq_strs = retokenize_and_verify(
            task_type,
            self.tokenizer,
            prompt_ids=prompt_ids,
            seq_ids=seq_ids,
            query_ids=[
                str(data_id) for data_id in data.ids for _ in range(self.group_size)
            ],
            check_xml_format=self.check_xml_format,
        )
        scores = torch.FloatTensor(format_rewards).to(packed_input_ids.device)
        scores[scores == 0] = -1

        scores = (
            scores.to(packed_input_ids.device) - self.output_bias
        ) * self.output_scaling

        self.log_rewards_to_file(task_type, model, prompt_strs, seq_strs, scores)

        res = SequenceSample(
            keys=["rewards"],
            trailing_shapes=dict(rewards=()),
            dtypes=dict(rewards=torch.float32),
            ids=data.ids,
            seqlens=dict(
                rewards=[
                    [1 for _ in range(len(x))] for x in data.seqlens["packed_input_ids"]
                ],
            ),
            data=dict(rewards=scores),
        )

        # record rewards for each piece of data
        avg_scores = []
        offset = 0
        for i in range(data.bs):
            score_lis = scores[
                offset : offset + len(data.seqlens["packed_input_ids"][i])
            ]
            avg_scores.append(score_lis.mean().item())
            offset += len(data.seqlens["packed_input_ids"][i])
        assert offset == sum(len(x) for x in data.seqlens["packed_input_ids"])

        res.metadata["scores"] = avg_scores

        if self.check_verifier_status:
            avg_score = torch.tensor(
                np.mean(avg_scores), device=constants.current_device()
            )
            dist.all_reduce(
                avg_score, op=dist.ReduceOp.SUM, group=constants.data_parallel_group()
            )
            avg_score /= constants.data_parallel_group()
            avg_score = avg_score.item()
            minimal_score = (-1 - self.output_bias) * self.output_scaling

            if avg_score <= minimal_score or np.isclose(avg_score, minimal_score):
                raise VerifierException(
                    "All rewards are at minimal value. Probably there are something wrong with the verifier!"
                )
        return res

    def log_rewards_to_file(
        self, task_type: str, model: model_api.Model, prompt_strs, seq_strs, scores
    ):
        tik = time.perf_counter()
        gen_file_path = os.path.join(
            self.answer_save_path,
            task_type,
            f"v{model.version.global_step}r{dist.get_rank()}.txt",
        )

        os.makedirs(os.path.dirname(gen_file_path), exist_ok=True)
        with open(gen_file_path, "w") as _f:
            for idx, (score, prompt_str, seq_str) in enumerate(
                zip(scores, prompt_strs, seq_strs)
            ):
                info = "\n".join(
                    [
                        f"idx: {idx} / {len(scores)}",
                        f"reward is {score.item()}, prompt is {colorama.Fore.YELLOW + colorama.Style.DIM}{prompt_str}{colorama.Style.RESET_ALL}",
                        f"sequence is: {colorama.Fore.YELLOW + colorama.Style.DIM}{seq_str.split(prompt_str)[1]}{colorama.Style.RESET_ALL}.",
                    ]
                )
                _f.write(info + "\n")

        gen_file_path = os.path.join(
            self.answer_save_path,
            task_type,
            f"v{model.version.global_step}r{dist.get_rank()}.jsonl",
        )
        os.makedirs(os.path.dirname(gen_file_path), exist_ok=True)
        with open(gen_file_path, "w") as _f:
            for idx, (score, prompt_str, seq_str) in enumerate(
                zip(scores, prompt_strs, seq_strs)
            ):
                _f.write(
                    json.dumps(
                        {
                            "prompt": prompt_str,
                            "generated": seq_str.split(prompt_str)[1],
                            "reward": score.item(),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        logger.info(f"[{task_type}] number of samples: {len(scores)}, {scores.shape}")
        logger.info(f"[{task_type}] avg reward: {sum(scores) / len(scores)}")
        logger.info(f"[{task_type}] log to file time: {time.perf_counter()- tik:.2f}s")

    def inference(
        self,
        model: model_api.Model,
        data: SequenceSample,
        mb_spec: MicroBatchSpec,
    ) -> SequenceSample | None:
        input_ = data
        data, backward_indices = self._dispatch_tp_and_pp(data)
        task_data, dispatch_indices = self._dispatch_tasks(data)

        assert self.rw_type == "sparse"

        def _task_func(func, task_type: str):
            def _wrapped_func(*args, **kwargs):
                start_time = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                except Exception as e:
                    raise asyncio.CancelledError(
                        f"[{task_type}] task failed: {e}"
                    ) from e
                finally:
                    duration = time.perf_counter() - start_time
                    logger.info(f"[{task_type}] time cost: {duration:.4f}s")
                return task_type, result

            return _wrapped_func

        async def _run_tasks():
            tasks = []
            for task_type, d in task_data.items():
                task_func = _task_func(self.calculate_task_reward, task_type)
                task_args = (model, d, mb_spec, task_type)
                task = asyncio.create_task(asyncio.to_thread(task_func, *task_args))
                tasks.append(task)

            results = await asyncio.gather(*tasks)
            task_results = {}
            for res in results:
                task_type, result = res
                task_results[task_type] = result

            return task_results

        def run_in_thread():
            # Create a new event loop for this thread
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(_run_tasks())
            finally:
                new_loop.close()

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor() as executor:
            future = executor.submit(run_in_thread)
            task_results = future.result()
        final_result = self._gather_tasks(task_results, dispatch_indices, data.bs)
        final_result = self._gather_tp_and_pp(input_, final_result, backward_indices)

        model.inc_version()

        return final_result

    def _mock_inference(
        self,
        model: model_api.Model,
        data: SequenceSample,
    ) -> SequenceSample:

        prompt_lens = flat2d(data.seqlens["packed_prompts"])
        task_ids = data.data["task_ids"].cpu().numpy().tolist()
        seqlens = []
        offset = 0
        seq = []
        for plen, task_id in zip(prompt_lens, task_ids):
            seq += [data.data["packed_prompts"][offset : offset + plen]]
            offset += plen
            if task_id == RL_TASKS.index("math"):
                answer_str = (
                    "something unimportant but the answer is \\boxed{-\\frac{2}{3}}."
                )
            elif task_id == RL_TASKS.index("code"):
                answer_str = (
                    "```python\ninput()\nimport time\ntime.sleep(1e-3)\nprint(1)\n```"
                )
            else:
                answer_str = "something unimportant"
            encoding = model.tokenizer(
                [answer_str], add_special_tokens=True, return_attention_mask=False
            )

            ans = torch.tensor(encoding["input_ids"], dtype=torch.long).flatten()
            seq += [ans]
            seqlens.append(plen + len(ans))

        x = SequenceSample.from_default(
            seqlens=seqlens,
            ids=data.ids,
            data=dict(packed_input_ids=torch.cat(seq)),
        )
        data.update_(x)
        return data


model_api.register_interface("rw-math-code", MultiTaskRewardInterface)
