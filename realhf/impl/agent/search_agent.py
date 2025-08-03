# Copyright 2025 Ant Group Inc.
# Licensed under the Apache License, Version 2.0 (the "License").

import asyncio
import copy
import json
import os
import time
from datetime import datetime
from typing import List, Dict

import colorama
import numpy as np
import torch

from realhf.api.core.agent_api import Agent, register_agent
from realhf.api.core.data_api import SequenceSample, load_hf_tokenizer
from realhf.api.core.env_api import EnvironmentService
from realhf.api.core.model_api import BundledGenerationOutputs
from realhf.base import constants, logging

logger = logging.getLogger("Search Agent")

def correct_format(idx, s):
    correct = all(
        [
            s.count("<search>") == s.count("</search>"),
            s.count("<information>") == s.count("</information>") == s.count("<|begin_of_documents|>") == s.count("<|end_of_documents|>") == 0,
            s.count("Assistant") == s.count("assistant") == 0,
            s.count("<think>") + int(idx == 0) == s.count("</think>"),
            (s.strip().endswith("</search>") or s.strip().endswith("</answer>")),
            s.count("</think>") > 0,
        ]
    )
    return correct

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
        remove_duplicate_docs: bool = False,
        topk: int = 5,
        prompt_type: str = "v0",
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

        assert self.gconfig.n == 1

        self.stop = ["</search>", "</search>\n", "</search>\n\n"]

        self.gconfig.stop = self.stop

        self.max_turns = max_turns
        self.filter_no_search_correct = filter_no_search_correct
        self.remove_duplicate_docs = remove_duplicate_docs
        self.topk = topk

        self.prompt_type = prompt_type
    
    @property
    def total_requests(self):
        return self.n

    async def collect_trajectory(self, prompt: SequenceSample, env: EnvironmentService, obs_queue: asyncio.Queue, act_queue: asyncio.Queue) -> List[SequenceSample]:
        await env.reset()

        assert prompt.bs == 1
        prompt_token_ids = prompt.data["packed_prompts"].cpu().numpy().tolist()
        qid = prompt.ids[0]

        
        prompt_str = self.tokenizer.batch_decode(
            [prompt_token_ids],
            clean_up_tokenization_spaces=False,
            skip_special_tokens=True,
        )[0]


        birth_time = int(datetime.now().timestamp() * 1000)
        start_time = last_time = time.time()

        parts: List[List[Dict]] = [[] for _ in range(self.n)]
        rewards = [0. for _ in range(self.n)]
        running = [True for _ in range(self.n)]
        submitted = [False for _  in range(self.n)]

        version_start = [None for _ in range(self.n)]
        version_end = [None for _ in range(self.n)]

        is_areal = self.prompt_type in ["v0"]

        while any(running):
            submit_time0=time.time()
            for i in range(self.n):
                if (not running[i]) or submitted[i]:
                    continue
                tokens = copy.deepcopy(prompt_token_ids)
                for part in parts[i]:
                    tokens = tokens + list(part["tokens"])
                if len(tokens) >= self.gconfig.max_new_tokens:
                    running[i] = False
                else:
                    _gconfig = self.gconfig.new(max_new_tokens = self.gconfig.max_new_tokens - len(tokens) + len(prompt_token_ids))
                    await obs_queue.put((f"{qid}##{i}-Part{len(parts[i])}", tokens, _gconfig))
                    submitted[i] = True
            submit_time1=time.time()

            act_time0=time.time()
            act: BundledGenerationOutputs = await act_queue.get()
            act_time1=time.time()

            i = eval(act.qid.split("##")[1].split("-")[0])
            assert running[i] and submitted[i]
            submitted[i] = False

            if version_start[i] is None:
                version_start[i] = act.version_start[0]
            version_end[i] = act.version_end[0]

            part = dict(tokens = act.output_ids[0], type="act", act = act)
            parts[i].append(part)

            tokens = [] # prompt_token_ids
            for part in parts[i]:
                tokens = tokens + list(part["tokens"])
            
            history_str, act_str = self.tokenizer.batch_decode(
                [tokens, act.output_ids[0]],
                clean_up_tokenization_spaces=False,
                skip_special_tokens=True,
            )
            parts[i][-1].update(dict(text=act_str))

            env_time0 = time.time()
            result = (await env.step((qid, [history_str])))[0]
            env_time1 = time.time()

            for k in ["score", "ground_truth"]:
                if k in result and result[k] is not None:
                    parts[i][-1].update({k: result[k]})
            
            # stop when eos & the answer is outputed or length limit is reached
            finish = (act.output_ids[0][-1] in self.eos and result["extracted"] is not None) or act.no_eos[0]

            if not finish and "documents" in result and result["documents"] is not None:
                documents = result["documents"]
                documents = [(k, doc) for k, doc in enumerate(documents)]
                if self.remove_duplicate_docs:
                    all_docs = sum([p["documents"] for p in parts[i] if p["type"] == "documents"], [])
                    documents = [(k, doc) for k, doc in documents if doc not in all_docs]
                documents = documents[:self.topk]
                if len(documents) > 0:
                    doc_id_template = "({doc_id})"
                    if self.prompt_type.startswith("feimo"):
                        doc_id_template = "Doc {doc_id}: "
                    doc_str = "\n\n<information>" + "\n\n".join([doc_id_template.format(doc_id=str(k+1)) + doc for k, doc in documents]) + "</information>\n\n"
                elif is_areal:
                    doc_str = "\n\n<information>\nNo More New Information is Found\n</information>\n\n"
                doc_token_ids = self.tokenizer(doc_str, add_special_tokens=False)["input_ids"]
                documents = [doc for _, doc in documents]
                parts[i].append(dict(type="documents", text=doc_str, tokens=doc_token_ids, documents=documents))
        
            # if the model outputs eos but no valid answer is find, this might be caused by incorrect answer format. 
            num_turns = len([p for p in parts[i] if p["type"] == "act"])

            if self.prompt_type.startswith("feimo"):
                if act.output_ids[0][-1] in self.eos:
                    finish = True
                if num_turns >= self.max_turns:
                    finish = True

            if is_areal:
                if act.output_ids[0][-1] in self.eos and result["extracted"] is None:
                    aux_str = "\n\n<think>\nI have tried to finish reasoning but the answer checker does not receive an parsable answer. The reason may be I have not report my answer or I did not put the answer correctly inside <answer></answer>.\n</think>\n\n"
                    aux_token_ids = self.tokenizer(aux_str, add_special_tokens=False)["input_ids"]
                    parts[i].append(dict(type="auxilliary", text=aux_str, tokens=aux_token_ids,))
                
                if num_turns >= self.max_turns:
                    if num_turns == self.max_turns:
                        aux_str = "\n\n<think>\nI have I did not put the answer correctly inside <answer></answer>.\n</think>\n\n"
                        aux_token_ids = self.tokenizer(aux_str, add_special_tokens=False)["input_ids"]
                        parts[i].append(dict(type="auxilliary", text=aux_str, tokens=aux_token_ids,))
                    else:
                        finish = True

            
            if finish:
                running[i] = False
                rewards[i] = result["score"]
            
            current_time = time.time()
            logger.info("Round time stats: qid={}. Total={}s. Round={}s. Submit={}s. Act={}s. Env={}s. Running={}. #Turns={}. #Retrivals={}".format(qid, current_time - start_time, current_time - last_time, submit_time1-submit_time0, act_time1-act_time0, env_time1-env_time0, running, [len([p for p in parts[i] if p["type"] == "act"]) for i in range(self.n)], [len([p for p in parts[i] if p["type"] == "documents"]) for i in range(self.n)]))
            last_time = current_time
    
        # put a finish signal
        await obs_queue.put((qid, None, self.gconfig))

        avg_r = np.mean([float(s) for s in rewards])
        max_r = max([float(s) for s in rewards])
        raw_rewards = rewards
        rewards = [ 
            ((float(r) - 0.5) * 2 - self.reward_bias) * self.reward_scaling
            for r in rewards
        ]
        format_rewards = [float(all([correct_format(j, p["text"]) for j, p in enumerate(parts[i]) if p["type"] == "act"])) for i in range(self.n)]
        rewards = [reward + self.format_reward * format_reward for reward, format_reward in zip(rewards, format_rewards)]

        #if avg_r < self.success_rate_lb:
        #    logger.info(f"Query ID {qid} reward too low: {avg_r} < {self.success_rate_lb}.")
        #    return []
        '''if max_r < self.success_rate_lb:
            logger.info(f"Query ID {qid} max reward too low: {max_r} < {self.success_rate_lb}.")
            return []'''
        if avg_r > self.success_rate_ub:
            logger.info(
                f"Query ID {qid} reward too high: {avg_r} > {self.success_rate_ub}."
            )
            return []
        
        if self.filter_no_search_correct and any(raw_rewards[i] > 0 and len([p for p in parts[i] if p["type"] == "documents"]) == 0 for i in range(self.n)):
            return []

        self.log_to_file(
            str(qid),
            prompt_str,
            parts=parts,
            rewards=raw_rewards,
            format_rewards=format_rewards,
            version_start=version_start,
            version_end=version_end,
        )

        # format parts into training data
        packed_input_ids = []
        prompt_mask = []
        packed_logprobs = []
        seq_no_eos_mask = []
        packed_prompts = prompt_token_ids
        seqlens = []
        
        for i in range(self.n):
            tokens_i = copy.deepcopy(prompt_token_ids)
            prompt_mask_i = [1] * (len(prompt_token_ids))
            logprobs_i = [0.] * (len(prompt_token_ids) - 1)
            no_eos_mask_i = None
            for part in parts[i]:
                part_len = len(part["tokens"])
                tokens_i = tokens_i + list(part["tokens"])
                if part["type"] == "act":
                    act = part["act"]
                    prompt_mask_i.extend([0.] * part_len)
                    logprobs_i.extend(list(act.logprobs[0])[-part_len:])
                    no_eos_mask_i = part["act"].no_eos[0]
                    assert (len(act.logprobs[0]) >= part_len), (qid, i, len(act.logprobs[0]), part_len)
                    # print("[DEBUG] make batch", i, len(act.logprobs[0]), len(act.output_ids[0]), len(part["tokens"]), len(tokens_i), flush=True)
                else:
                    prompt_mask_i.extend([1.] * part_len)
                    logprobs_i.extend([0.] *(part_len))
            
            seqlens.append(len(tokens_i))
            packed_input_ids.extend(tokens_i)
            prompt_mask.extend(prompt_mask_i)
            packed_logprobs.extend(logprobs_i)
            seq_no_eos_mask.append(no_eos_mask_i)

            assert (len(logprobs_i) == seqlens[-1] -1 ), (len(logprobs_i), seqlens[-1] -1)
        
        # logging stats
        logging = []
        logging_keys = ["num_queries", "act_tokens", "doc_tokens", "seqlen", "format_reward"]
        for i in range(self.n):
            for k in logging_keys:
                v = None
                if k == "num_queries":
                    v = len([p for p in parts[i] if p["type"] == "documents"])
                elif k == "act_tokens":
                    v = sum([len(p["tokens"]) for p in parts[i] if p["type"] == "act"])
                elif k == "doc_tokens":
                    v = sum([len(p["tokens"]) for p in parts[i] if p["type"] == "documents"])
                elif k == "seqlen":
                    v = sum([len(p["tokens"]) for p in parts[i]])
                elif k == "format_reward":
                    v = format_rewards[i]
                logging.append(v)

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
                seq_no_eos_mask=[[1 for _ in range(self.n)]],
                rewards=[[1 for _ in range(self.n)]],
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
                rewards=torch.tensor(rewards, dtype=torch.float32),
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
    
    def log_to_file(
        self,
        qid,
        prompt_str,
        parts,
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
            for j, part in enumerate(parts[i]):
                d = dict(group_idx=i, part_idx=j, prompt=prompt_str, version_start=version_start[i], version_end=version_end[i], final_reward=rewards[i], 
                format_reward=format_rewards[i],
                correct_format = correct_format(j, part["text"]) if part["type"] == "act" else None,
                n_tokens=len(part["tokens"]), **part)
                if "act" in d:
                    d.pop("act")
                monitor_file.write(
                    json.dumps(d, ensure_ascii=False) + "\n"
                )

register_agent("search", SearchAgent)