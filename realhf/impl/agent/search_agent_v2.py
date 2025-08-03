# Copyright 2025 Ant Group Inc.
# Licensed under the Apache License, Version 2.0 (the "License").

import asyncio
import copy
import json
import os
import time
from datetime import datetime
from dataclasses import asdict
from typing import List, Dict, Optional

import colorama
import numpy as np
import torch

from realhf.api.core.agent_api import Agent, register_agent
from realhf.api.core.data_api import SequenceSample, load_hf_tokenizer
from realhf.api.core.env_api import EnvironmentService
from realhf.api.core.model_api import BundledGenerationOutputs
from realhf.base import constants, logging
from realhf.impl.agent.reasoning_agent import AReaLSearchQwen3AgentV1

logger = logging.getLogger("Search Agent")

import psutil
import time
from datetime import datetime
import logging

def get_memory_usage():
    memory = psutil.virtual_memory()
    return {
        'total': memory.total,
        'available': memory.available,
        'used': memory.used,
        'percent': memory.percent,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }


def correct_format(idx, s):
    correct = all(
        [
            s.count("<search>") == s.count("</search>"),
            s.count("<access>") == s.count("</access>"),
            s.count("<answer>") == s.count("</answer>"),
            # s.count("<information>") == s.count("</information>") == s.count("<|begin_of_documents|>") == s.count("<|end_of_documents|>") == 0,
            s.count("Assistant") == s.count("assistant") == 0,
            s.count("</think>") <= 1,
           #  (s.strip().endswith("</search>") or s.strip().endswith("</answer>") or s.strip().endswith("</access>") or s.strip().endswith("</think>")),
        ]
    )
    return correct

class SearchTrajectory:
    parts: List[Dict]

    def __init__(self, parts):
        self.parts = parts

    def compose_training_trajs(self, prompt_length, max_new_tokens):
        trajs = []
        for part in self.parts:
            tokens_i = list(part["prompt_token_ids"]) + list(part["token_ids"])
            prompt_mask_i = [1] * len(part["prompt_token_ids"]) + [0] * len(part["token_ids"])
            logprobs_i = [0.] * (len(part["prompt_token_ids"]) - 1) + list(part["logprobs"])[-len(part["token_ids"]):]
            no_eos_mask_i = part["no_eos"]
            
            if len(tokens_i) > max_new_tokens + prompt_length:
                logger.info(f"This trajectory is reduced since it is too long: {prompt_length} + {max_new_tokens} < {len(tokens_i)}")
                reduce_length = len(tokens_i) - max_new_tokens + prompt_length
                tokens_i = tokens_i[:-reduce_length]
                prompt_mask_i = prompt_mask_i[:-reduce_length]
                logprobs_i = logprobs_i[:-reduce_length]

            trajs.append((tokens_i, prompt_mask_i, logprobs_i, no_eos_mask_i))

        return trajs

class SearchAgent(Agent):
    def __init__(
        self,
        gconfig,
        tokenizer_path,
        success_rate_lb,
        success_rate_ub,
        reward_scaling=1.0,
        reward_bias=0.0,
        max_turns: int = 100,
        n: int =1,
        format_reward: float = 0.,
        filter_no_search_correct: bool = False,
        reward_type: str = "F1", # "EM" / "CEM" / "Max"
        topk: int = 5,
        prompt_type: str = "v1",
        group_baseline: bool = False,
        cut_ratio_decay: str ="none",
        search_result_cut_len: Optional[int] = None,
        llm_as_judge: bool = False,
        valid_inst_ratio: float = 0.0,
    ):
        self.n=n
        self.gconfig = gconfig
        self.tokenizer = load_hf_tokenizer(tokenizer_path)
        self.eos: List[int] = self.tokenizer("<|im_end|>", "<|endoftext|>")["input_ids"]

        self.success_rate_lb = success_rate_lb
        self.success_rate_ub = success_rate_ub

        self.format_reward = format_reward
        self.reward_scaling = reward_scaling
        self.reward_bias = reward_bias
        self.reward_type = reward_type
        self.group_baseline = group_baseline
        self.cut_ratio_decay = cut_ratio_decay
        self.search_result_cut_len = search_result_cut_len
        self.llm_as_judge = llm_as_judge
        self.valid_inst_ratio = valid_inst_ratio

        assert self.gconfig.n == 1

        self.stop = ["<|end_of_text|>", "<|im_end|>"]
        # self.gconfig.stop = self.stop

        self.max_turns = max_turns
        self.topk = topk

        self.prompt_type = prompt_type

    async def collect_trajectory(self, prompt: SequenceSample, env: EnvironmentService, obs_queue: asyncio.Queue, act_queue: asyncio.Queue) -> List[SequenceSample]:
        await env.reset()

        is_valid_inst = np.random.uniform(0, 1) < self.valid_inst_ratio

        # assert self.n == 1
        assert prompt.bs == 1
        prompt_token_ids = prompt.data["packed_prompts"].cpu().numpy().tolist()
        qid = prompt.ids[0]

        prompt_str = self.tokenizer.batch_decode(
            [prompt_token_ids],
            clean_up_tokenization_spaces=False,
            skip_special_tokens=True,
        )[0]
        
        # skip calc
        if "calc" in qid:
            await obs_queue.put((qid, None, self.gconfig))
            return []

        hist_r = self.load_history_rewards(qid)
        if hist_r is not None:
            logger.info(f"Skip {qid} since this question is also trained on")
            await obs_queue.put((qid, None, self.gconfig))
            return []


        birth_time = int(datetime.now().timestamp() * 1000)
        start_time = last_time = time.time()

        trajs: List[SearchTrajectory] = [SearchTrajectory([]) for _ in range(self.n)]
        rewards = [0. for _ in range(self.n)]

        version_start = [None for _ in range(self.n)]
        version_end = [None for _ in range(self.n)]

        processes = [dict(id=f"{qid}-{i}", question=prompt_str, prompt=prompt_str, running=True, submitted=False) for i in range(self.n)]
        reasoning_agent = AReaLSearchQwen3AgentV1(max_turns=self.max_turns, topk=self.topk, force_valid=is_valid_inst)

        
        queries: List[Dict] = [None for _ in range(self.n)]
        while any([process["running"] for process in processes]):
            logger.info(f"Running status @ Qid={qid}: running={[process['running'] for process in processes]}. submitted={[process['submitted'] for process in processes]}. ")
            submit_time0=time.time()
            for i, process in enumerate(processes):
                if process["running"] and not process["submitted"]:
                    process_queries = reasoning_agent.prepare_queries(self.tokenizer, [process])
                    if len(process_queries) == 0:
                        process["running"] = False
                    else:
                        process["submitted"] = True
                        assert len(process_queries) == 1
                        query = dict(process_id=i, **process_queries[0])
                        if query["type"] == "llm":
                            query["sampling"]["max_new_tokens"] = self.gconfig.max_new_tokens - query["query_len"] + len(prompt_token_ids)
                            tokens = self.tokenizer([query["prompt"]], add_special_tokens=False)["input_ids"][0][:len(prompt_token_ids) + self.gconfig.max_new_tokens]
                            _gconfig = self.gconfig.new(n=1, max_new_tokens = max(0, self.gconfig.max_new_tokens - len(tokens) + len(prompt_token_ids)))
                            # _gconfig.stop = self.stop
                            logger.info(f"generation request for {qid}##{i}-Step{len(process['history'])}: {json.dumps(asdict(_gconfig))}")
                            await obs_queue.put((f"{qid}##{i}-Step{len(process['history'])}", tokens, _gconfig))
                        queries[i] = query
            submit_time1=time.time()
            logger.info(f"Submission status @ Qid={qid}: running={[process['running'] for process in processes]}. submitted={[process['submitted'] for process in processes]}. ")
            if all([not process["submitted"] for process in processes]):
                logger.info(f"No submitted query for qid {qid}")
                await asyncio.sleep(0.1)
                continue
            submit_time1=time.time()

            # handle search and url access
            # Note only the first step would handle multiple search/access queries
            env_time0 = time.time()
            for i in range(self.n):
                if queries[i] is not None and queries[i]["type"] in ["search", "access"]:
                    process, query = processes[i], queries[i]
                    query_str = f"<{query['type']}>" + query.get("query", query.get("urls", [""]))[0] + f"</{query['type']}>"
                    result = (await env.step((qid, [query_str])))[0]
                    if "server_type" not in result:
                        result["server_type"] = "train"
                    rewards[i] = result["score"]
                    reasoning_agent.consume_responses([process], [query], [result])
                    process["submitted"] = False
                    queries[i] = None
            env_time1 = time.time()
        
            if not any([query is not None and query["type"] == "llm" for query in queries]):
                continue

            # get one llm generation output
            act_time0=time.time()
            act: BundledGenerationOutputs = await act_queue.get()
            act_time1=time.time()
            
            i = eval(act.qid.split("##")[1].split("-")[0])
            process = processes[i]
            assert process["running"] and process["submitted"]
            process["submitted"] = False

            logger.info(f"Handling {act.qid}. input length={len(act.prompt_ids)}. output length={len(act.output_ids[0])}, no_eos={act.no_eos}")
            if version_start[i] is None:
                version_start[i] = act.version_start[0]
            version_end[i] = act.version_end[0]

            act_str = self.tokenizer.batch_decode(
                [act.output_ids[0]],
                clean_up_tokenization_spaces=False,
                skip_special_tokens=True,
            )[0]

            trajs[i].parts.append(dict(prompt_token_ids=act.prompt_ids, token_ids=act.output_ids[0], output_str=act_str, logprobs=act.logprobs[0], no_eos=act.no_eos[0]))

            reasoning_agent.consume_responses([process], [queries[i]], [act_str])
            queries[i] = None
            
            current_time = time.time()

            logger.info("Round time stats: qid={}. Total={}s. Round={}s. Submit={}s. Act={}s. Env={}s. Running={}. #Turns={}. #Retrivals={}".format(act.qid, current_time - start_time, current_time - last_time, submit_time1-submit_time0, act_time1-act_time0, env_time1-env_time0, [p["running"] for p in processes], [len([p for p in processes[i]["history"] if p["type"] == "act"]) for i in range(self.n)], ["{}/{}".format(len([p for p in processes[i]["history"] if p["type"] == "documents"]), len([p for p in processes[i]["history"] if p["type"] == "page"])) for i in range(self.n)]))

            mem = get_memory_usage()
            logger.info(
                f"qid={act.qid}, "
                f"Memory Usage - Total: {mem['total']/1024/1024:.2f} MB, "
                f"Used: {mem['used']/1024/1024:.2f} MB ({mem['percent']}%), "
                f"Available: {mem['available']/1024/1024:.2f} MB"
            )
            
            
            last_time = current_time
            asyncio.sleep(0.1)

        # put a finish signal
        await obs_queue.put((qid, None, self.gconfig))

        if self.llm_as_judge:
            start = time.time()
            logger.info("Qid {} start LLM-as-Judge".format(qid))
            for i in range(self.n):
                rewards[i] = (await env.eval_llm_judge((qid, reasoning_agent.answers([processes[i]]))))[0]
                processes[i]["MBE"] = rewards[i]
                await asyncio.sleep(0.1)
            end = time.time()
            logger.info("Qid {} LLM-as-Judge time: {:.3f}. LLM-as-Judge results: {}".format(qid, end-start, rewards))


        avg_r = np.mean([float(s) for s in rewards])
        max_r = max([float(s) for s in rewards])
        raw_rewards = rewards
        rewards = [ 
            ((float(r) - 0.5) * 2 - self.reward_bias) * self.reward_scaling
            for r in rewards
        ]
        # rewards = [reward + self.format_reward * format_reward for reward, format_reward in zip(rewards, format_rewards)]
        
        _rewards = []
        _valid_inst = []
        _judge_q_invalid_err = []
        for i, (raw_r, r) in enumerate(zip(raw_rewards, rewards)):
            pred_ans = reasoning_agent.answers([processes[i]])[0]
            judge_q_invalid = (await env.judge_q_invalid((qid, reasoning_agent.answers([processes[i]]))))[0]

            reward = ((float(raw_r) - 0.5) * 2 - self.reward_bias) * self.reward_scaling #  * format_r
            if is_valid_inst and judge_q_invalid:
                reward = (-1 - self.reward_bias - 2) * self.reward_scaling

            _rewards.append(reward)
            _valid_inst.append(is_valid_inst)
            _judge_q_invalid_err.append(judge_q_invalid and is_valid_inst)
        
        rewards = _rewards
        
        self.log_to_file(
            str(qid),
            prompt_str,
            processes=processes,
            rewards=raw_rewards,
            version_start=version_start,
            version_end=version_end,
        )
        
        #if avg_r < self.success_rate_lb:
        #    logger.info(f"Query ID {qid} reward too low: {avg_r} < {self.success_rate_lb}.")
        #    return []
        if max_r < self.success_rate_lb:
            logger.info(f"Query ID {qid} max reward too low: {max_r} < {self.success_rate_lb}.")
            return []
        if avg_r > self.success_rate_ub:
            logger.info(
                f"Query ID {qid} reward too high: {avg_r} > {self.success_rate_ub}."
            )
            return []
        
        # logging stats
        logging = []
        logging_keys = ["num_queries", "num_accesses", "num_valid_accesses", "act_tokens", "reward", "no_eos", "valid_inst", "judge_invalid_err"]
        for i in range(self.n):
            process = processes[i]
            for k in logging_keys:
                v = None
                if k == "num_queries":
                    v = len([p for p in process["history"] if p["type"] == "documents"])
                elif k == "num_accesses":
                    v = len([p for p in process["history"] if p["type"] == "page" and "Page 1" in p["info_str"]])
                elif k == "num_valid_accesses":
                    v = len([p for p in process["history"] if p["type"] == "page" and "No More Info" not in p["info_str"] and "Page 1" in p["info_str"]])
                elif k == "act_tokens":
                    v = sum([len(p["token_ids"]) for p in trajs[i].parts])
                elif k == "reward":
                    v = rewards[i]
                elif k == "no_eos":
                    v = any([p["no_eos"] for p in trajs[i].parts])
                elif k == "valid_inst":
                    v = _valid_inst[i]
                elif k == "judge_invalid_err":
                    v = _judge_q_invalid_err[i]
                logging.append(v)
            

        if self.group_baseline:
            avg_reward = sum(rewards) / len(rewards)
            rewards = [r - avg_reward for r in rewards]

            if all([r==0 for r in rewards]):
                logger.info(
                    f"Query ID {qid} all rewards are the same."
                )
                return []

        # format parts into training data
        packed_input_ids = []
        prompt_mask = []
        packed_logprobs = []
        seq_no_eos_mask = []
        packed_prompts = prompt_token_ids
        seqlens = []
        final_rewards = []
        
        for i in range(self.n):
            results = trajs[i].compose_training_trajs(len(prompt_token_ids), self.gconfig.max_new_tokens)
            for res in results:
                if res is not None:
                    tokens_ij, prompt_mask_ij, logprobs_ij, no_eos_mask_ij = res

                    seqlens.append(len(tokens_ij))
                    packed_input_ids.extend(tokens_ij)
                    prompt_mask.extend(prompt_mask_ij)
                    packed_logprobs.extend(logprobs_ij)
                    seq_no_eos_mask.append(no_eos_mask_ij)
                    final_rewards.append(rewards[i])

                    assert (len(logprobs_ij) == seqlens[-1] -1 ), (len(logprobs_ij), seqlens[-1] -1)
        

        x = SequenceSample(
            keys=[
                "packed_input_ids",
                "prompt_mask",
                "packed_logprobs",
                "seq_no_eos_mask",
                "packed_prompts",
                "version_start",
                "version_end",
                "rewards",
                "birth_time",
                "logging",
            ],
            ids=[qid],
            dtypes=dict(
                packed_prompts=torch.long,
                packed_input_ids=torch.long,
                prompt_mask=torch.bool,
                seq_no_eos_mask=torch.bool,
                version_start=torch.int,
                version_end=torch.int,
                packed_logprobs=torch.float32,
                rewards=torch.float32,
                birth_time=torch.long,
                logging=torch.float32,
            ),
            trailing_shapes=dict(
                packed_input_ids=(),
                prompt_mask=(),
                seq_no_eos_mask=(),
                packed_prompts=(),
                version_end=(),
                version_start=(),
                packed_logprobs=(),
                rewards=(),
                birth_time=(),
                logging=(),
            ),
            seqlens=dict(
                packed_input_ids=[seqlens],
                packed_logprobs=[[s - 1 for s in seqlens]],
                packed_prompts=[[len(prompt_token_ids)]],
                prompt_mask=[seqlens],
                seq_no_eos_mask=[[1 for _ in seqlens]],
                rewards=[[1 for _ in seqlens]],
                version_start=[[1 for _ in range(self.n)]],
                version_end=[[1 for _ in range(self.n)]],
                birth_time=[[1]],
                logging=[[len(logging_keys) for _ in range(self.n)]]
            ),
            data=dict(
                packed_prompts=torch.tensor(packed_prompts, dtype=torch.long),
                packed_logprobs=torch.tensor(packed_logprobs, dtype=torch.float32),
                packed_input_ids=torch.tensor(packed_input_ids, dtype=torch.long),
                seq_no_eos_mask=torch.tensor(seq_no_eos_mask, dtype=torch.bool),
                rewards=torch.tensor(final_rewards, dtype=torch.float32),
                version_start=torch.tensor(version_start, dtype=torch.int),
                version_end=torch.tensor(version_end, dtype=torch.int),
                birth_time=torch.tensor([birth_time], dtype=torch.long),
                prompt_mask=torch.tensor(prompt_mask, dtype=torch.bool),
                logging=torch.tensor(logging, dtype=torch.float32,),
            ),
        )
        y = SequenceSample(
            keys=["task_ids"],
            ids=[qid],
            dtypes=dict(task_ids=torch.long),
            trailing_shapes=dict(task_ids=()),
            seqlens=dict(task_ids=[[1]]),
            data=dict(task_ids=torch.tensor([4], dtype=torch.long)),
        )
        x.update_(y)

        return [x]
    
    def load_history_rewards(self, qid):
        import glob
        train_monitor_file_path = os.path.join(
            constants.LOG_ROOT,
            constants.experiment_name(),
            constants.trial_name(),
            "training_monitor",
            "version",
            f"{qid}.jsonl",
        )
        train_monitor_file_path = train_monitor_file_path.replace("version", "*")
        filenames=glob.glob(train_monitor_file_path)
        rewards = []
        for filename in filenames:
            trajs=[json.loads(ff) for ff in open(filename)]
            if "final_reward" not in trajs[0]:
                continue
            avg_r = np.mean([traj["final_reward"] for traj in trajs])
            rewards.append(avg_r)
        if len(rewards) == 0:
            return None
        return np.mean(rewards)

    def log_to_file(
        self,
        qid,
        prompt_str,
        processes,
        rewards,
        version_start,
        version_end,
    ):
        train_monitor_file_path = os.path.join(
            constants.LOG_ROOT,
            constants.experiment_name(),
            constants.trial_name(),
            "training_monitor",
            str(max(version_end)),
            f"{qid}.jsonl",
        )
        os.makedirs(os.path.dirname(train_monitor_file_path), exist_ok=True)
        
        monitor_file = open(train_monitor_file_path, "a")

        for i in range(self.n):
            processes[i]["prompt"] = prompt_str
            processes[i]["traj_idx"] = i
            processes[i]["final_reward"] = rewards[i]
            monitor_file.write(
                json.dumps(processes[i], ensure_ascii=False) + "\n"
            )

register_agent("search-reasoning-v1", SearchAgent)