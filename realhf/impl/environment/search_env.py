# Copyright 2025 Ant Group Inc.

import asyncio
import os
import re
import random
import httpx
from typing import List, Tuple

import asyncio
import aiohttp

from aiohttp.client import ClientTimeout
from dataclasses import asdict
from openai import AsyncOpenAI

from realhf.api.core.env_api import EnvironmentService, register_environment
from realhf.base import logging, name_resolve, names
from realhf.impl.dataset.search_dataset import load_metadata
from realhf.impl.environment.search_qa_em import compute_score_em, compute_score_f1, cover_exact_match_score_1

import asyncio
import time
try:
    from realhf.impl.environment.online_search import AsyncOnlineSearchServer
    from realhf.impl.environment.online_web_browser import SimpleTextBrowser, WebPageInfo
except:
    print("Cannot import online search.")


logger = logging.getLogger("Search Environment")

def get_server_list():
    name_root = names.rag_retrieval_server()
    server_list = name_resolve.get_subtree(name_root)
    return server_list

async def post_request(server_addr, req_meta):
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"http://{server_addr}/retrieve",
            json=req_meta,
            timeout=ClientTimeout(total=120, sock_connect=120),
        ) as response:
            response.raise_for_status()
            res = await response.json()
            return res

async def test_post_request():
    for addr in server_list:
        server_addr = addr # "10.11.22.184:5217"
        req_meta = {
            "queries": ["China"],
            "topk": 5,
            "return_scores": False
        }
        res = post_request(addr, req_meta)
        print(addr, res.keys())
    

def is_valid_html(s):
    pass

class SearchEnv(EnvironmentService):
    def __init__(self, dataset_path: str, reward_type: str = "F1", topk:int = 10, online_search: bool = False, online_url_access: bool = False, llm_as_judge: bool = False, use_jina=False, jina_api_key=None):
        self.id2info = load_metadata(dataset_path)
        self.server_list = get_server_list()
        self.server_addr = None if len(self.server_list) == 0 else random.choice(self.server_list)
        self.reward_type = reward_type
        self.llm_as_judge = llm_as_judge
        self.topk = topk

        logger.info(f"Search Engine at {self.server_addr}")

        # online search
        self.online_search = online_search
        self.online_url_access = online_url_access
        self.use_jina = use_jina
        self.jina_api_key = jina_api_key

        if self.online_search or self.online_url_access:
            # self.search_server = AsyncOnlineSearchServer(use_jina=self.use_jina, jina_api_key=self.jina_api_key)
            self.search_server = AsyncOnlineSearchServer()
        
        if self.llm_as_judge:
            self.llm_servers = self.get_llm_server_list()
            self.llm_server = random.choice(self.llm_servers)
            self.llm_client = AsyncOpenAI(base_url=f"http://{self.llm_server}/v1", api_key="None", timeout=httpx.Timeout(180.0))
            logger.info(f"LLM as Judge will be used. Connected to llm server @ {self.llm_server}")
        
            self.llm_judge_lock = asyncio.Lock()
    
    def get_llm_server_list(self):
        # for llm-as-judge
        import etcd3
        REAL_ETCD_ADDR="etcd-client.openpsi-etcd.svc.sigma-na130-lingbo.na130.wl-robby.local:2379"
        host,port=REAL_ETCD_ADDR.split(":")
        name_root=f"admin/gjx-serving/qwen2.5-72b-inst"
        self.etcd3_client = client = etcd3.client(host=host,port=port,user=None, password=None)
        server_list = list(client.get_prefix(name_root))
        server_list = [x[0].decode("utf-8") for x in server_list]
        logger.info(f"Qwen2.5-72B-Instruct servers: {server_list}")
        return server_list
    
    async def chat_with_llm_server(self, llm_client, prompt: str, max_tokens: int = 2048, max_retries: int = 16):
        for retry_idx in range(max_retries):
            try:
                # Create a chat completion request
                response = await llm_client.chat.completions.create(
                    model="default",  # Use "default" or your specific model name
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=0.6,
                    stream=False  # Set to True if you want streaming responses
                )
                
                return response.choices[0].message.content
            except Exception as e:
                logger.warning(f"Error communicating with SGLang service: {e}")
                # self.llm_servers = [l for l in self.llm_servers if l!=self.llm_server]
                llm_server = random.choice(self.llm_servers)
                self.llm_client = llm_client = AsyncOpenAI(base_url=f"http://{llm_server}/v1", api_key="None", timeout=httpx.Timeout(60.0))
                logger.warning(f"Retry chating with llm server: {retry_idx+1}/{max_retries}. Switch llm server to {llm_server}")
                await asyncio.sleep(10.0)
        return ""
    
    async def eval_llm_judge(self, qid_answers: Tuple[str, List[str]]):
        async with self.llm_judge_lock:
            prompt_template = "You are an evaluation assistant. Please determine if the predicted answer is equivalent to the labeled answer.\n" \
            "You should first give your rationale for the judgement, and then give your judgement result (i.e., correct or incorrect).\n\n" \
            "\n" \
            "question: {question}\n" \
            "ground truth answers: {gt_answer}\n" \
            "pred_answer: {pred_answer}\n\n" \
            "Did the model give an answer **equivalent** to the labeled answer? \n\nThe output should in the following json format:\n" \
            "```json\n" \
            "{{\n" \
            """    "rationale": "your rationale for the judgement, as a text",\n""" \
            """    "judgement": "your judgement result, can only be 'correct' or 'incorrect'\n""" \
            "}}\n" \
            "```\n" \
            "Your output:" 
            qid, answers = qid_answers
            logger.info(f"Qid {qid} acquired llm judge lock.")
            data = self.id2info[qid.split("@")[0]]
            ground_truth = data["answer"]
            question = data["question"]
            results = []

            if isinstance(ground_truth, list) and len(ground_truth) == 1:
                ground_truth = str(ground_truth[0])

            llm_server = random.choice(self.llm_servers)
            llm_client = self.llm_client 
            '''AsyncOpenAI(base_url=f"http://{llm_server}/v1", api_key="None", timeout=httpx.Timeout(180.0))'''

            for ans in answers:
                # llm-as-judge prompt
                prompt = prompt_template.format(question=question, gt_answer=str(ground_truth), pred_answer=ans[:200])

                # parse response
                raw_response = await self.chat_with_llm_server(llm_client, prompt, max_tokens=8192)

                # parse results
                import json, ast
                mbe = None
                for parse_fn in [json.loads, ast.literal_eval]:
                    try:
                        mbe = parse_fn(raw_response.split("```json")[-1].split("```")[0].strip())
                        break
                    except:
                        print(f"[WARNING] Error parsing {[raw_response]}")
                if mbe is None and '"judgement": "incorrect"' in raw_response:
                    mbe = dict(judgement="incorrect")
                if mbe is None and '"judgement": "correct"' in raw_response:
                    mbe = dict(judgement="correct")
                if mbe is None:
                    logger.info(f"Unknown judge result: {[raw_response]}")
                    mbe = dict(judgement="unknown")
                
                logger.info("LLM as Judge for Qid={}. GT={}. Ans={}. Result: MBE={}. Raw Response={}".format(qid, ground_truth, ans, json.dumps(mbe), raw_response[:500]))

                score = float("judgement" in mbe and mbe["judgement"] == "correct")
                results.append(score)
            return results
    
    async def judge_q_invalid(self, qid_answers: Tuple[str, List[str]]):
        async with self.llm_judge_lock:
            prompt_template = "You will be given a question and a model-generated answer. You need to judge whether the model-generated answer claims that the question is invalid.\n" \
            "You should first give your rationale for the judgement, and then give your judgement result (i.e., yes or no).\n" \
            "\n" \
            "question: {question}\n" \
            "model-generated answer: {pred_answer}\n\n" \
            "The output should in the following json format:\n" \
            "```json\n" \
            "{{\n" \
            """    "rationale": "your rationale for the judgement, as a text",\n""" \
            """    "judgement": "your judgement result, can only be 'yes' or 'no'\n""" \
            "}}\n" \
            "```\n" \
            "Your output:" 
            qid, answers = qid_answers
            logger.info(f"Qid {qid} acquired llm judge lock for judge_q_invalid.")
            data = self.id2info[qid.split("@")[0]]
            question = data["question"]
            results = []

            llm_server = random.choice(self.llm_servers)
            llm_client = self.llm_client 

            for ans in answers:
                # llm-as-judge prompt
                prompt = prompt_template.format(question=question, pred_answer=ans[:200])

                # parse response
                raw_response = await self.chat_with_llm_server(llm_client, prompt, max_tokens=8192)

                # parse results
                import json, ast
                mbe = None
                for parse_fn in [json.loads, ast.literal_eval]:
                    try:
                        mbe = parse_fn(raw_response.split("```json")[-1].split("```")[0].strip())
                        break
                    except:
                        print(f"[WARNING] Error parsing {[raw_response]}")
                if mbe is None and '"judgement": "yes"' in raw_response:
                    mbe = dict(judgement="yes")
                if mbe is None and '"judgement": "no"' in raw_response:
                    mbe = dict(judgement="no")
                if mbe is None:
                    logger.info(f"Unknown judge result: {[raw_response]}")
                    mbe = dict(judgement="unknown")
                
                logger.info("Judge Q Invalid for Qid={}. Ans={}. Result: MBE={}. Raw Response={}".format(qid, ans, json.dumps(mbe), raw_response[:500]))

                score = float("judgement" in mbe and mbe["judgement"] == "yes")
                results.append(score)
            return results

    async def reset(self, seed=None, options=None):
        return None, {}
    
    async def post_request(self, req_meta, _type="retrieve"):
        # local rag server
        cnt = 0
        while cnt < 10:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"http://{self.server_addr}/{_type}",
                        json=req_meta,
                        timeout=ClientTimeout(total=120, sock_connect=120),
                    ) as response:
                        response.raise_for_status()
                        res = await response.json()
                        return res
            except:
                self.server_list = get_server_list()
                self.server_addr = random.choice(self.server_list)
                logger.info(f"Search Engine switched to {self.server_addr}")
                cnt += 1
                await asyncio.sleep(10)
                # pass
        raise RuntimeError("Search Engines are not available")

    def process_webpage(self, content):
        keys = [("title", "title"), ("p", "p"), ("li", "li", lambda c: "\n" not in c), ("td", "td"), ("tr", "tr")] 
        content_list = []
        init_length = len(content)
        while any([f"<{k[0]}" in content and f"</{k[1]}>" in content for k in keys]):
            klr = []
            for k in keys:
                start = 0
                # print(k)
                while True:
                    ls = [content[start:].find(f"<{k[0]}{c}") for c in [">", " "]]
                    ls = [l for l in ls if l != -1]
                    l = -1 if len(ls) == 0 else min(ls)
                    # print(ls)
                    if l == -1:
                        break
                    l += start
                    r = content[l:].find(f"</{k[1]}>")
                    if r == -1:
                        break
                    if (len(k) <= 2) or (len(k) >= 3 and k[2](content[l:l+r])):
                        # print(k, l, l+r)
                        klr.append((k, l, l+r))
                        break
                    start = l + r

            if len(klr) == 0:
                break
            klr = sorted(klr, key=lambda x:x[1])
            k, l, r = klr[0]
            content_list.append(content[l:r+len(f"</{k[1]}>")])
            # print(content_list[-1])
            # input("stop...")
            if k[0] == "p":
                content_list[-1] += "\n\n"
            elif k[0] == "li":
                content_list[-1] += "\n"
            content = content[r:]
        content = "".join(content_list)
        final_length = len(content)
        logger.info(f"process the webpage: {init_length} -> {final_length}. {content[:100]}")
        return content

    async def step(self, qid_actions: Tuple[str, List[str]]):
        qid, actions = qid_actions

        results = []
        for action in actions:
            result = dict(documents=None, score=None, ground_truth=None, type=None)
            if "<search>" in action and "</search>" in action:
                query = action.split("<search>")[-1].split("</search>")[0]
                req_meta = {
                    "queries": [query],
                    "topk": self.topk,
                    "return_scores": False
                }
                documents, urls = [], []

                if self.online_search or self.id2info[qid.split("@")[0]].get("source", "N/A") == "WebWalkerQA":
                    response = await self.search_server.query_async(req_meta)
                    response[0] = response[0][:5]
                    random.shuffle(response[0])
                    online_urls = [r["link"] for r in response[0]]
                    online_snippets = ["Google Search Result: " + r.get("title", "") + " " +  r["snippet"].strip() for r in response[0]]
                    urls.extend(online_urls)
                    documents.extend(online_snippets)
                else:
                    res = await self.post_request(req_meta, _type="retrieve")
                    res["result"][0] = res["result"][0][:5]
                    random.shuffle(res["result"][0])
                    documents = [r["contents"] for r in res["result"][0]]
                    urls = [r["url"] for r in res["result"][0]]

                result["documents"] = documents
                result["urls"] = urls
                result["type"] = "search"
            elif "<access>" in action and "</access>" in action:
                url = action.split("<access>")[-1].split("</access>")[0].strip().replace("_", "%20").replace("https://en.wikipedia.org/wiki/", "https://en.wikipedia.org/w/index.php/")
                req_meta = {
                    "urls": [url],
                }
                '''res = await self.post_request(req_meta, _type="access")
                if res["result"][0] is None:
                    page = None
                else:
                    page = res["result"][0]["contents"]'''
                page = None

                if page is None and self.online_url_access:
                    response = await self.search_server.access_async([url])
                    if self.use_jina:
                        page = response[0].get("page", "")
                    else:
                        # process webpage
                        page = self.process_webpage(response[0].get("page", ""))
                else:
                    res = await self.post_request(req_meta, _type="access")
                    if res["result"][0] is None:
                        page = None
                    else:
                        page = res["result"][0]["contents"]
            
                result["page"] = page
                result["type"] = "access"
            
            if "<|answer_split|>" in self.id2info[qid.split("@")[0]]["answer"]:
                self.id2info[qid.split("@")[0]]["answer"] = self.id2info[qid.split("@")[0]]["answer"].split("<|answer_split|>")

            ground_truth = self.id2info[qid.split("@")[0]]["answer"]
            if isinstance(ground_truth, list) or isinstance(ground_truth, tuple):
                ground_truth = [str(gt) for gt in ground_truth]
            else:
                ground_truth = str(ground_truth)

            ground_truth_aug = None
            if "aug_answer" in self.id2info[qid.split("@")[0]]:
                ground_truth_aug = self.id2info[qid.split("@")[0]]["aug_answer"]
                if isinstance(ground_truth_aug, list) or isinstance(ground_truth_aug, tuple):
                    ground_truth_aug = [str(gt) for gt in ground_truth_aug]
                else:
                    ground_truth_aug = str(ground_truth_aug)
            
            if self.reward_type == "F1":
                extracted, score = compute_score_f1(action, ground_truth, method="strict")
            elif self.reward_type == "EM":
                extracted, score = compute_score_em(action, ground_truth, method="strict")
            if ground_truth_aug is not None:
                if self.reward_type == "F1":
                    _, score_aug = compute_score_f1(action, ground_truth_aug, method="strict")
                elif self.reward_type == "EM":
                    _, score_aug = compute_score_em(action, ground_truth_aug, method="strict")

            result["score"] = score
            result["ground_truth"] = self.id2info[qid.split("@")[0]]["answer"]
            result["extracted"] = extracted
            if ground_truth_aug is not None:
                score_aug = max(score_aug, score)
                result["score"] = score * 0.7 + score_aug * 0.3
                result["ground_truth_aug"] = ground_truth_aug
            if extracted is not None:
                logger.info("F1 Score={:.2f}. Extracted='{}'. Ground Truth='{}'. Qid={}. Question='{}'".format(score, extracted, ground_truth, qid.split("@")[0], self.id2info[qid.split("@")[0]]["question"]))
            # if score > 0:
            #     logger.info(f"Correct. extracted='{extracted}' ground_truth='{ground_truth}'")
            # else:
            #     logger.info(f"Wrong. extracted='{extracted}' ground_truth='{ground_truth}'")
            results.append(result)
        return results

register_environment("search", SearchEnv)
