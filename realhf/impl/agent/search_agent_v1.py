# Copyright 2025 Ant Group Inc.
# Licensed under the Apache License, Version 2.0 (the "License").

import asyncio
import copy
import json
import os
import time
from datetime import datetime
from typing import List, Dict, Optional

import colorama
import numpy as np
import torch

from realhf.api.core.agent_api import Agent, register_agent
from realhf.api.core.data_api import SequenceSample, load_hf_tokenizer
from realhf.api.core.env_api import EnvironmentService
from realhf.api.core.model_api import BundledGenerationOutputs
from realhf.base import constants, logging

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
    running: bool
    paused: bool 

    def __init__(self, parts, running=True):
        self.parts = parts
        self.running = running
        self.paused = False

    def copy(self):
        return SearchTrajectory(parts=[p for p  in self.parts], running = self.running)
    
    def compose_training_traj(self, prompt_token_ids, max_new_tokens):
        tokens_i = copy.deepcopy(prompt_token_ids)
        prompt_mask_i = [1] * (len(prompt_token_ids))
        logprobs_i = [0.] * (len(prompt_token_ids) - 1)
        no_eos_mask_i = None
        for part in self.parts:
            part_len = len(part["tokens"])
            tokens_i = tokens_i + list(part["tokens"])
            if part["type"] == "act":
                act = part["act"]
                prompt_mask_i.extend([0.] * part_len)
                logprobs_i.extend(list(act.logprobs[0])[-part_len:])
                no_eos_mask_i = part["act"].no_eos[0]
                assert (len(act.logprobs[0]) >= part_len), (len(act.logprobs[0]), part_len)
            else:
                prompt_mask_i.extend([1.] * part_len)
                logprobs_i.extend([0.] *(part_len))
        if no_eos_mask_i is None:
            # no act
            return None
        if len(tokens_i) > max_new_tokens + len(prompt_token_ids):
            logger.info(f"This trajectory is reduced since it is too long: {len(prompt_token_ids)} + {max_new_tokens} < {len(tokens_i)}")
            reduce_length = len(tokens_i) - max_new_tokens + len(prompt_token_ids)
            tokens_i = tokens_i[:-reduce_length]
            prompt_mask_i = prompt_mask_i[:-reduce_length]
            logprobs_i = logprobs_i[:-reduce_length]
        return tokens_i, prompt_mask_i, logprobs_i, no_eos_mask_i

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

        assert self.gconfig.n == 1

        def all_stop_keys(c):
            ret = []
            for l in [" ", "\n"]:
                for r in [" ", "\n"]:
                    for nl in range(0,3+1):
                        for nr in range(0, 3+1):
                            _k = (l * nl) + c + (r * nr)
                            if _k not in ret:
                                ret.append(_k)
            return ret
        self.stop = all_stop_keys("</search>") + all_stop_keys("</access>") + ["</answer>"]
        self.stop_think = ["</think>", "</think>\n", "</think>\n\n", "</think>\n\n\n"]
        self.think_word = "<think>"

        self.gconfig.stop = self.stop

        self.max_turns = max_turns
        self.filter_no_search_correct = filter_no_search_correct
        self.topk = topk

        self.prompt_type = prompt_type

    async def collect_trajectory(self, prompt: SequenceSample, env: EnvironmentService, obs_queue: asyncio.Queue, act_queue: asyncio.Queue) -> List[SequenceSample]:
        await env.reset()

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


        birth_time = int(datetime.now().timestamp() * 1000)
        start_time = last_time = time.time()

        trajs: List[List[SearchTrajectory]] = [[SearchTrajectory([])] for _ in range(self.n)]
        rewards = [0. for _ in range(self.n)]
        n_steps = [0 for _ in range(self.n)]
        running = [True for _ in range(self.n)]
        submitted = [False for _ in range(self.n)]

        version_start = [None for _ in range(self.n)]
        version_end = [None for _ in range(self.n)]
        page_cache = [None for _ in range(self.n)]

        is_areal = self.prompt_type in ["v0"]

        while any(running):
            submit_time0=time.time()
            for i in range(self.n):
                if (not running[i]) or submitted[i]:
                    continue
                for traj_idx, traj in enumerate(trajs[i]):
                    if not traj.running or traj.paused:
                        continue

                    tokens = copy.deepcopy(prompt_token_ids)
                    for part in traj.parts:
                        tokens = tokens + list(part["tokens"])

                    if len(tokens) >= self.gconfig.max_new_tokens:
                        logger.info(f"{qid}-{i}. Traj_idx={traj_idx}. too long.")
                        if traj_idx == 0:
                            running[i] = False
                        else:
                            trajs[i][traj_idx].running = False
                            trajs[i][0].paused = False
                    else:
                        _gconfig = self.gconfig.new(max_new_tokens = self.gconfig.max_new_tokens - len(tokens) + len(prompt_token_ids))
                        if traj_idx == 0:
                            _gconfig.stop = self.stop
                        else:
                            _gconfig.stop = self.stop_think
                        await obs_queue.put((f"{qid}##{i}-{traj_idx}-Step{n_steps[i]}", tokens, _gconfig))
                        n_steps[i] += 1
                        submitted[i] = True
            if not any(submitted):
                continue
            assert (any(submitted) or any(any([traj.running and not traj.paused for traj in trajs[i]]) for i in range(self.n) if running[i])), (submitted, running, [[(traj.running, traj.paused) for traj in trajs[i]] for i in range(self.n)])
            submit_time1=time.time()

            act_time0=time.time()
            act: BundledGenerationOutputs = await act_queue.get()
            act_time1=time.time()

            i = eval(act.qid.split("##")[1].split("-")[0])
            traj_idx = eval(act.qid.split("##")[1].split("-")[1])
            assert running[i] and submitted[i]
            submitted[i] = False

            logger.info(f"Handling {act.qid}")

            if version_start[i] is None:
                version_start[i] = act.version_start[0]
            version_end[i] = act.version_end[0]

            part = dict(tokens = act.output_ids[0], type="act", act = act)
            trajs[i][traj_idx].parts.append(part)

            tokens = [] # prompt_token_ids
            for part in  trajs[i][traj_idx].parts:
                tokens = tokens + list(part["tokens"])
            
            history_str, act_str = self.tokenizer.batch_decode(
                [tokens, act.output_ids[0]],
                clean_up_tokenization_spaces=False,
                skip_special_tokens=True,
            )
            trajs[i][traj_idx].parts[-1].update(dict(text=act_str))
            
            if traj_idx == 0:
                env_time0 = time.time()
                result = (await env.step((qid, [act_str])))[0]
                env_time1 = time.time()

                for k in ["score", "ground_truth"]:
                    if k in result and result[k] is not None:
                        trajs[i][traj_idx].parts[-1].update({k: result[k]})
            
                # stop when eos & the answer is outputed or length limit is reached
                finish = (act.output_ids[0][-1] in self.eos and result["extracted"] is not None) or act.no_eos[0] or ("<answer>" in act_str and "</answer>" in act_str)

                if not finish and result["type"] == "search" and result["documents"] is not None:
                    documents = result["documents"]
                    urls = result["urls"]
                    urls = urls[:self.topk]
                    documents = documents[:self.topk]

                    if len(documents) > 0:
                        cut_len = 5000 
                        if self.search_result_cut_len is not None:
                            cut_len = self.search_result_cut_len
                            if self.cut_ratio_decay != "none":
                                _cur_version = act.version_end[0]
                                cut_len = max(100, int(eval(self.cut_ratio_decay.format(version=_cur_version))))
                        doc_id_template = "[Doc {doc_id}]({url}):\n"
                        def random_crop(s, l):
                            if len(s) <= l:
                                return s
                            start = np.random.randint(0, len(s) - l + 1)
                            return s[start:][:l].strip()
                        doc_str = "\n\n<information>\n" + "\n\n".join([doc_id_template.format(doc_id=str(k+1), url=url) + random_crop(doc, cut_len) for k, (doc, url) in enumerate(zip(documents, urls))]) + "\n</information>\n\n" + "<think>"
                        short_doc_str = "\n\n<information>" + "\n\n".join([doc_id_template.format(doc_id=str(k+1), url=url) + random_crop(doc, cut_len) + "..." for k, (doc, url) in enumerate(zip(documents, urls))]) + "\n</information>\n\n" + "<think>"

                        doc_token_ids = self.tokenizer(doc_str, add_special_tokens=False)["input_ids"]
                        short_doc_token_ids = self.tokenizer(short_doc_str, add_special_tokens=False)["input_ids"]
                        
                        trajs[i].append(trajs[i][0].copy())
                        trajs[i][0].paused = True
                        trajs[i][0].parts.append(dict(type="documents", text=short_doc_str, tokens=short_doc_token_ids, documents=documents))
                        trajs[i][-1].parts.append(dict(type="documents", text=doc_str, tokens=doc_token_ids, documents=documents))
                    elif is_areal:
                        doc_str =  "\n\n<information>\nNo More New Information is Found\n</information>\n\n" + "<think>"

                        doc_token_ids = self.tokenizer(doc_str, add_special_tokens=False)["input_ids"]
                        
                        trajs[i][0].parts.append(dict(type="documents", text=doc_str, tokens=doc_token_ids, documents=documents))
                    
                if not finish and result["type"] == "access":
                    page = result["page"]

                    if page is not None and page.strip() != "":
                        # put page into cache
                        page = page[:250000]
                        _page_str_length = len(page)
                        _page_token_length = len(self.tokenizer(page, add_special_tokens=False)["input_ids"])
                        page_cache[i] = []
                        while len(page) > 0 and len(page_cache[i]) < 10:
                            _len = min(25000, len(page))
                            page_cache[i].append(f">>>> Page {len(page_cache[i]) + 1} >>>>\n\n" + page[:_len])
                            page = page[_len:]
                        logger.info(f"Qid {act.qid} reading webpage w. str_length={_page_str_length} num_tokens={_page_token_length} num_chunks={len(page_cache[i])}")
                        
                        # read page
                        page = page_cache[i].pop(0)
                        logger.info(f"Qid {act.qid} reading page: {page.strip().split("\n")[0].strip()}")
                        page_str = "\n\n<information>" + page + "\n</information>\n\n" + "<think>"
                        short_page_str = "\n\n<information>\n" + page[:100] + "...\n\n" + "</information>\n\n" + "<think>"

                        page_token_ids = self.tokenizer(page_str, add_special_tokens=False)["input_ids"]
                        short_page_token_ids = self.tokenizer(short_page_str, add_special_tokens=False)["input_ids"]
                        
                        trajs[i].append(trajs[i][0].copy())
                        trajs[i][0].paused = True
                        trajs[i][0].parts.append(dict(type="page", text=short_page_str, tokens=short_page_token_ids, page=page))
                        trajs[i][-1].parts.append(dict(type="page", text=page_str, tokens=page_token_ids, page=page))
                    else:
                        page_str = "\n\n<information>\nNo More Information is Found for this URL.\n</information>\n\n" + "<think>"
                        page_token_ids = self.tokenizer(page_str, add_special_tokens=False)["input_ids"]
                        trajs[i][0].parts.append(dict(type="page", text=page_str, tokens=page_token_ids, page=page))
                
                if act.output_ids[0][-1] in self.eos and result["extracted"] is None:
                    aux_str = "\n\n<think>\nI have tried to finish reasoning but the answer checker does not receive an parsable answer. The reason may be I have not report my answer or I did not put the answer correctly inside <answer></answer>.\n</think>\n\n"
                    aux_token_ids = self.tokenizer(aux_str, add_special_tokens=False)["input_ids"]
                    trajs[i][0].parts.append(dict(type="auxilliary", text=aux_str, tokens=aux_token_ids,))
                    
                # if the model outputs eos but no valid answer is find, this might be caused by incorrect answer format. 
                num_turns = len([p for p in trajs[i][0].parts if p["type"] in ["act"]])

                if num_turns > self.max_turns:
                    finish = True

                if finish:
                    running[i] = False
                    rewards[i] = result["score"]
                    
                current_time = time.time()
                logger.info("Round time stats: qid={}. Total={}s. Round={}s. Submit={}s. Act={}s. Env={}s. Running={}. Traj Running={}. #Turns={}. #Retrivals={}".format(act.qid, current_time - start_time, current_time - last_time, submit_time1-submit_time0, act_time1-act_time0, env_time1-env_time0, running, [[(traj.running, traj.paused) for traj in trajs[i]] for i in range(self.n)], [len([p for p in trajs[i][0].parts if p["type"] == "act"]) for i in range(self.n)], ["{}/{}".format(len([p for p in trajs[i][0].parts if p["type"] == "documents"]), len([p for p in trajs[i][0].parts if p["type"] == "page"])) for i in range(self.n)]))
            else:
                assert (trajs[i][0].paused), (i, trajs[i][0].paused, [[(traj.running, traj.paused) for traj in trajs[i]] for i in range(self.n)])
                trajs[i][0].paused = False
                trajs[i][traj_idx].running = False

                part = dict(tokens = act.output_ids[0], text=act_str, type="act_other", act = act)
                trajs[i][0].parts.append(part)

                if page_cache[i] is not None and len(page_cache[i]) > 0:
                    # read page
                    page = page_cache[i].pop(0)
                    logger.info(f"Qid {act.qid} reading page: {page.strip().split("\n")[0].strip()}")
                    page_str = "\n\n<information>" + page + "\n</information>\n\n" + "<think>"
                    short_page_str = "\n\n<information>\n" + page[:100] + "...\n\n" + "</information>\n\n" + "<think>"

                    page_token_ids = self.tokenizer(page_str, add_special_tokens=False)["input_ids"]
                    short_page_token_ids = self.tokenizer(short_page_str, add_special_tokens=False)["input_ids"]
                    
                    trajs[i].append(trajs[i][0].copy())
                    trajs[i][0].paused = True
                    trajs[i][0].parts.append(dict(type="page", text=short_page_str, tokens=short_page_token_ids, page=page))
                    trajs[i][-1].parts.append(dict(type="page", text=page_str, tokens=page_token_ids, page=page))

                current_time = time.time()
                logger.info("Round time stats: qid={}. Total={}s. Round={}s. Submit={}s. Act={}s. Running={}. Traj Running={}. #Turns={}. #Retrivals={}".format(act.qid, current_time - start_time, current_time - last_time, submit_time1-submit_time0, act_time1-act_time0, running, [[(traj.running, traj.paused) for traj in trajs[i]] for i in range(self.n)], [len([p for p in trajs[i][0].parts if p["type"] == "act"]) for i in range(self.n)], ["{}/{}".format(len([p for p in trajs[i][0].parts if p["type"] == "documents"]), len([p for p in trajs[i][0].parts if p["type"] == "page"])) for i in range(self.n)]))
            
            mem = get_memory_usage()
            logger.info(
                f"qid={act.qid}, "
                f"Memory Usage - Total: {mem['total']/1024/1024:.2f} MB, "
                f"Used: {mem['used']/1024/1024:.2f} MB ({mem['percent']}%), "
                f"Available: {mem['available']/1024/1024:.2f} MB"
            )
            
            
            last_time = current_time
    
        # put a finish signal
        await obs_queue.put((qid, None, self.gconfig))

        # if max(rewards) >= 0.8:
        #     rewards = [0 if r < 0.6 else r for r in rewards]
        # else:
        #     rewards = [r for r in rewards]

        avg_r = np.mean([float(s) for s in rewards])
        max_r = max([float(s) for s in rewards])
        raw_rewards = rewards
        rewards = [ 
            ((float(r) - 0.5) * 2 - self.reward_bias) * self.reward_scaling
            for r in rewards
        ]
        format_rewards = [float(all([correct_format(j, p["text"]) for j, p in enumerate(trajs[i][0].parts) if p["type"] in ["act", "act_other"]])) for i in range(self.n)]
        # rewards = [reward + self.format_reward * format_reward for reward, format_reward in zip(rewards, format_rewards)]
        
        _rewards = []
        _valid_inst = []
        _judge_q_invalid_err = []
        for i, (raw_r, r, format_r) in enumerate(zip(raw_rewards, rewards, format_rewards)):
            is_valid_inst = ("Note: the question is a valid question" in prompt_str)
            tokens = [] # prompt_token_ids
            for part in trajs[i][0].parts:
                tokens = tokens + list(part["tokens"])
            history_str = self.tokenizer.batch_decode(
                [tokens],
                clean_up_tokenization_spaces=False,
                skip_special_tokens=True,
            )[0]
            pred = history_str.split("<answer>")[-1].split("</answer>")[0].strip()
            judge_q_invalid = any([_c in pred for _c in ["question", "invalid", "appropriate", "valid"]])

            reward = ((float(raw_r) * float(format_r) - 0.5) * 2 - self.reward_bias) * self.reward_scaling #  * format_r
            if is_valid_inst and judge_q_invalid:
                reward = (-1 - self.reward_bias - 0.2) * self.reward_scaling

            _rewards.append(reward)
            _valid_inst.append(is_valid_inst)
            _judge_q_invalid_err.append(judge_q_invalid and is_valid_inst)
        
        rewards = _rewards


        self.log_to_file(
            str(qid),
            prompt_str,
            trajs=trajs,
            rewards=raw_rewards,
            format_rewards=format_rewards,
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
        
        # if self.filter_no_search_correct and any(raw_rewards[i] > 0 and len([p for p in parts[i] if p["type"] == "documents"]) == 0 for i in range(self.n)):
        #     return []
        
        # logging stats
        logging = []
        logging_keys = ["num_queries", "num_accesses", "num_valid_accesses", "act_tokens", "doc_tokens", "page_tokens", "seqlen", "format_reward", "reward", "no_eos", "valid_inst", "judge_invalid_err"]
        for i in range(self.n):
            for k in logging_keys:
                v = None
                if k == "num_queries":
                    v = len([p for p in trajs[i][0].parts if p["type"] == "documents"])
                elif k == "num_accesses":
                    v = len([p for p in trajs[i][0].parts if p["type"] == "page"])
                elif k == "num_valid_accesses":
                    v = len([p for p in trajs[i][0].parts if p["type"] == "page" and p["page"] is not None])
                elif k == "act_tokens":
                    v = sum([len(p["tokens"]) for p in trajs[i][0].parts if p["type"] in ["act", "act_other"]])
                elif k == "doc_tokens":
                    v = sum([len(p["tokens"]) for traj in trajs[i] for p in traj.parts if p["type"] == "documents"])
                elif k == "page_tokens":
                    v = sum([len(p["tokens"]) for traj in trajs[i] for p in traj.parts if p["type"] == "page"])
                elif k == "seqlen":
                    v = sum([len(p["tokens"]) for p in trajs[i][0].parts])
                elif k == "format_reward":
                    v = format_rewards[i]
                elif k == "reward":
                    v = rewards[i]
                elif k == "no_eos":
                    v = [p["act"].no_eos[0] for p in trajs[i][0].parts if p["type"] in ["act", "act_other"]][-1]
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
            for traj_idx, traj in enumerate(trajs[i]):
                res = traj.compose_training_traj(prompt_token_ids, self.gconfig.max_new_tokens)
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
            trajs = [ds for ds in trajs if ds["traj_idx"] == 0]
            rewards.extend([ds["final_reward"][0] for ds in trajs])
        return 


    def log_to_file(
        self,
        qid,
        prompt_str,
        trajs,
        rewards,
        format_rewards,
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
            for j, traj in enumerate(trajs[i]):
                ds = dict(group_idx=i, traj_idx=j)
                for k, part in enumerate(traj.parts):
                    d = dict(part_idx=k, prompt=prompt_str, version_start=version_start[i], version_end=version_end[i], final_reward=rewards[i], 
                    format_reward=format_rewards[i],
                    correct_format = correct_format(j, part["text"]) if part["type"] == "act" else None,
                    n_tokens=len(part["tokens"]), **part)

                    if "act" in d:
                        d.pop("act")

                    for key in d.keys():
                        if key not in ds:
                            ds[key] = []
                        ds[key].append(d[key])
                        
                monitor_file.write(
                    json.dumps(ds, ensure_ascii=False) + "\n"
                )

register_agent("search-v1", SearchAgent)