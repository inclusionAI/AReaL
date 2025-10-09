import re
from typing import Dict, List, Any, Optional
from transformers import PreTrainedTokenizerFast, AutoProcessor
from constants import TOOL_CROP_SYSTEM_PROMPT
from areal.utils.image import get_multimodal_input_ids_len
from AVoyager.utils.reward import compute_score

class AReaLVOYAGEReasoningAgentV1:
    
    def __init__(self,
                 max_turns: int = 128,
                 force_turns: int = 4,
                 topk: int = 10,
                 force_valid: bool = True):

        self.max_turns = max_turns
        self.force_turns = force_turns
        self.force_valid = force_valid
        self.topk = topk
        # 保持与原agent相同的属性名
        self.stop = ["<|im_end|>", "<|endoftext|>"]
        self.stop_sequences = self.stop

        print(f"AReaLVOYAGEReasoningAgentV1 初始化完成")

    def get_query_from_text(self, text: str) -> Optional[str]:
        pattern = r'<grounding>(.*?)</grounding>'
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return "<grounding>" + matches[-1].strip() + "</grounding>"

        return None
      
    def get_thought_from_text(self, text: str) -> Optional[str]:
        pattern = r'<thought>(.*?)</thought>'
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return "<think>" + matches[-1].strip() + "</think>"
            # return "<think>" + matches[-1].strip() + "</think>"
        
        return None

    def get_answer_from_text(self, text: str) -> Optional[str]:
        pattern = r'<answer>(.*?)</answer>'
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return "<answer>" + matches[-1].strip() + "</answer>"
        
        return None

    def print_grounding_debug_info(self, text: str):
        query_starts = text.count('<grounding>')
        query_ends = text.count('</grounding>')
        # print(f"搜索标签统计: {query_starts}个开始标签, {query_ends}个结束标签")

    
    # def debug_generation_tags(self, text: str) -> Dict:
    #     tags = {
    #         'query': {'open': text.count('<begin_of_query|>'), 'close': text.count('<end_of_query|>')},
    #         'documents': {'open': text.count('<begin_of_documents|>'), 'close': text.count('<end_of_documents|>')},
    #         'answer': {'open': text.count('<answer>'), 'close': text.count('</answer>')}
    #     }
    #
    #     for tag_name, counts in tags.items():
    #         tags[tag_name]['balanced'] = counts['open'] == counts['close']  
    #     return tags

    def all_finished(self, processes: List[Dict]) -> bool:
        finished = []
        for process in processes:
            finished.append(not process.get("running", True))
        return all(finished)

    def prepare_queries(self, tokenizer, processes: List[Dict], processor: AutoProcessor = None) -> List[Dict]:
        #目前把history简化为question之后的内容，之后还需要拓展或者做if分支
        '''
        process:[{
            "id": "unique id",
            "question": "the question",
            "answer": "the answer (optional)",
            "images": [PIL images] (optional),
            "history": [
                {"type": "prompt", "text": initial prompt text},
                {"type": "act", "text": "<think>...</think>\n\n<grounding>...</grounding>"},
                {"type": "grounding", "text": "<grounding>...</grounding>"},
                ...
            ],
            "running": True/False,
            "phase": "grounding"/"tool_call"/"answer",
            "cache_gen_text": "...", # optional, for caching incomplete generation
            "llm_gen_fail": int, # optional, count of consecutive LLM generation failures
            "page_cache": [page contents], # optional, cache of pages to be processed
            ...
        }]
        '''
        queries = []
        for process in processes:
            if "history" not in process:
                assert "pred_answer" not in process
                process["history"] = [dict(type="prompt", text=process["prompt"])]
                process["running"] = True
                process["phase"] = "grounding"  
            
            if process["running"]:
                #上一轮为调用工具，这一轮要分析，目前不适用
                if "text" not in process["history"][-1] and "info_str" in process["history"][-1]:
                    pass
                
                #     history = ""
                #     for idx, h in enumerate(process["history"][:-1]):
                #         history += h.get("short_info_str", h.get("text", ""))
                #     if len(history) > 25000:
                #         history = history[-25000:]
                    
                #     if process["history"][-1]["type"] == "page":
                #         prompt = ASearcherReasoningPrompts.READ_PAGE_PROMPT.format(question=process
                #     else:
                #         raise RuntimeError(f"Not supported history type: {process['history'][-1]['type']}")
                    
                #     input_text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
                #     query_len = tokenizer([input_text], return_length=True)['length'][0]

                #     if query_len <= 28000:
                #         print(f"Reading @ Qid {process['id']}", len(tokenizer(input_text, add_special_tokens=False)["input_ids"]), len([h for h in process["history"] if h["type"] == "documents"]), len([h for h in process["history"] if h["type"] == "act"]), flush=True)
                #         queries.append(dict(
                #             type="llm",
                #             sampling=dict(stop=self.stop, max_new_tokens=31000-query_len),
                #             query_len=query_len,
                #             prompt=prompt, 
                #         ))
                #         continue
                    
                #     if "cache_gen_text" in process:
                #         process.pop("cache_gen_text")
                
                #上一轮回答了判断有没有工具调用
                if "text" in process["history"][-1]:
                    last_text = process["history"][-1]["text"]
                    if ("<grounding>" in last_text and 
                        last_text.strip().endswith("</grounding>")):
                        if True:
                            query_text = last_text.split("<grounding>")[-1].split("</grounding>")[0].strip()
                            queries.append(dict(
                                type="grounding", 
                                query=[query_text.strip()], 
                                search_params=dict(topk=self.topk),
                                images=process.get("images", []),
                            ))
                            continue
                
                
                #初始情形，使用初始prompt
                # input_text = "".join([h["text"] for h in process["history"]])
                history = ""
                for idx, h in enumerate(process["history"]):
                    history += h.get("short_info_str", h.get("text", ""))
                if len(history) > 25000:
                    history = history[-25000:]

                if "images" in process and len(process["images"]) > 0 and processor is not None:
                    messages = [{"role": "system", "content": TOOL_CROP_SYSTEM_PROMPT}]
                    messages.append({"role": "user", "content": process["question"]})
                    messages.append({"role": "assistant", "content": history})
                else:
                    messages = [{"role": "user", "content": process["question"] + "\n\n" + history}]
                # prompt = ASearcherReasoningPrompts.THINK_AND_ACT_PROMPT.format(question=process["question"], history=history)
                input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False) + process.get("cache_gen_text", "")
                # input_text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False) + process.get("cache_gen_text", "")
                # print(f"Generate Act for Qid {process['id']}", len(tokenizer(input_text, add_special_tokens=False)["input_ids"]), len([h for h in process["history"] if h["type"] == "documents"]), len([h for h in process["history"] if h["type"] == "act"]), flush=True)

                #超过轮数或者token数，直接回答
                if any([
                    len([h for h in process["history"] if h["type"] == "grounding"]) >= 20,
                    len([h for h in process["history"] if h["type"] == "act"]) >= self.force_turns,
                    process.get("phase", "tool_call") == "answer",
                    ]):
                    process["phase"] = "answer"
                    print(f"Direct Generate Answer for Qid {process['id']}", len(tokenizer(input_text, add_special_tokens=False)["input_ids"]), len([h for h in process["history"] if h["type"] == "act"]), flush=True)
                    if "images" in process and len(process["images"]) > 0 and processor is not None:
                        messages = [{"role": "system", "content": TOOL_CROP_SYSTEM_PROMPT}]
                        messages.append({"role": "user", "content": process["question"]})
                        messages.append({"role": "assistant", "content": history})
                    else:
                        messages = [{"role": "user", "content": process["question"] + "\n\n" + history}]
                # if self.force_valid:
                    # prompt = prompt.replace('4. If you find information contradicting context of the question, you should point out that the question is invalid and the incorrect information in the question.', "4. You should find the most likely answer even when conflicting information is founded.")
                input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False) + process.get("cache_gen_text", "")

                # print("Query Input Length (llm):", process["id"], len(tokenizer(input_text, add_special_tokens=False)["input_ids"]),  len([h for h in process["history"] if h["type"] == "documents"]), len([h for h in process["history"] if h["type"] == "act"]), flush=True)
                if get_multimodal_input_ids_len(text=process["question"], tokenizer=tokenizer, images=process.get("images"), processor=processor) > 32000 or self.get_answer_from_text(process["history"][-1].get("text", "")) is not None:
                    print("process is done (1)", process["id"])
                    process["running"] = False
                    continue
                
                query_len = get_multimodal_input_ids_len(text=input_text, tokenizer=tokenizer, images=process.get("images"), processor=processor)
                process["max_new_tokens"] = max(0, 31000 - query_len)
                queries.append(dict(
                    type="llm", 
                    sampling=dict(stop=self.stop, max_new_tokens=process.get("max_new_tokens", 4096)),
                    query_len=query_len,
                    prompt=input_text, 
                    images=process.get("images", [])
                ))
                process.pop("max_new_tokens")
        
        return queries

    def consume_responses(self, processes: List[Dict], queries: List[Dict], responses: List[Any]) -> List[Dict]:       
        '''
        processes: 最开始的输入
        queries: prepare_queries的输出
        responses: 对应queries的输出
        ''' 
        i = 0
        for process in processes:
            if process["running"]:
                q, r = queries[i], responses[i]

                # print("consume response", process["id"], q["type"])
                #上一轮调用工具，这一轮要分析
                if q["type"] == "grounding":
                    if isinstance(r, list) and len(r) == 1:
                        r = r[0]
                    if r["status"] == "success":
                        process["history"].append(dict(
                            type="grounding", 
                            info_str=r["text"],
                            short_info_str=r["text"],
                            image=r["image"],
                        ))
                    else:
                        process["history"].append(dict(
                            type="grounding", 
                            info_str=r["text"],
                            short_info_str=r["text"],
                            image=None,
                        ))
               #上一轮没有调用工具，目前只可能是结束
                elif q["type"] == "llm":
                    if hasattr(r, 'stop_reason') and hasattr(r, 'text'):
                        generated_text = r.text
                    elif isinstance(r, dict):
                        generated_text = r.get('text', str(r))
                    else:
                        generated_text = r

                    if generated_text is None:
                        generated_text = ""
                    
                    raw_generated_text = generated_text
                    generated_text = process.get("cache_gen_text", "") + generated_text


                    self.print_grounding_debug_info(generated_text)
                    
                    extracted_thought = self.get_thought_from_text(generated_text)
                    extracted_answer = self.get_answer_from_text(generated_text)
                    extracted_query = self.get_query_from_text(generated_text)


                    # if the prompt is not asking to answer
                    if "<answer>" not in q["prompt"] and extracted_answer is not None:
                        print(f"Not time for producing answer for {process['id']}", extracted_answer, flush=True)
                        extracted_answer = None
                    
                    think_and_act = ""
                    if extracted_thought is not None:
                        think_and_act = think_and_act + extracted_thought
                    for act in [extracted_query, extracted_answer]:
                        if act is not None:
                            think_and_act = think_and_act.strip() + "\n\n" + act
                            break
                    
                    ### print(">>> THINK & ACT >>>\n", think_and_act, flush=True)

                    if extracted_thought is not None:
                        process["history"].append(dict(
                            type="act", 
                            full_reasoning_text = generated_text,
                            text=think_and_act.strip()
                        ))
                        if "cache_gen_text" in process:
                            process.pop("cache_gen_text")
                            
                    elif len(raw_generated_text) == 0:
                        process["cache_gen_text"] = ""
                        process["llm_gen_fail"] = process.get("llm_gen_fail", 0) + 1
                        if process["llm_gen_fail"] > 32:
                            print("process is done (2)", process["id"], process["llm_gen_fail"])
                            process["running"] = False
                    else:
                        if process["history"][-1]["type"] in ["grounding"]:
                            process["cache_gen_text"] = ""
                            process["history"].append(dict(
                                type="act", 
                                full_reasoning_text = generated_text,
                                text="<think>\n\n</think>"
                            ))
                            process["llm_gen_fail"] = process.get("llm_gen_fail", 0) + 1
                            process["page_cache"] = []
                        else:
                            process["cache_gen_text"] = generated_text
                        # process["max_new_tokens"] = process.get("max_new_tokens", 2048) + 1024
                    action_count = len([h for h in process["history"] if h["type"] == "act"])
                    if action_count >= self.max_turns + 20 or "<answer>" in think_and_act:
                        print("process is done (3)", process["id"], action_count, self.max_turns, "<answer>" in think_and_act, flush=True)
                        process["running"] = False

                # print("[DEBUG]  history length", process["id"], process["history"][-1]["type"], len(process["history"]), len(process.get("page_cache", [])), "page_cache" in process, len([h for h in process["history"] if h["type"] == "act"]))

                
                i += 1
        
        return processes

    def answers(self, processes: List[Dict]) -> List[str]:

        answers = []
        for process in processes:
            if "pred_answer" not in process:
                full_text = "".join(
                    [h["text"] for h in process["history"] if h["type"] != "prompt" and "text" in h]
                )
                
                if "<answer>" in full_text and "</answer>" in full_text:
                    answer = full_text.split("<answer>")[-1].split("</answer>")[0].strip()
                else:
                    reasoning_text = "\n\n".join([h["full_reasoning_text"] for h in process["history"] if "full_reasoning_text" in h] + [process.get("cache_gen_text", "")])
                    # find the last line metioning 'answer'
                    lines = reasoning_text.split("\n")
                    lines = [l for l in lines if 'answer' in l.lower()]
                    if len(lines) > 0:
                        answer = lines[-1]
                    else:
                        answer = reasoning_text.strip().split("</think>")[-1].strip()
                
                process["pred_answer"] = answer
            
            answers.append(process["pred_answer"])
        
        return answers

from areal.experimental.openai import ArealOpenAI

def parse_judge_result(raw_response):
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
        print(f"[WARNING] Unknown judge result: {[raw_response]}")
        mbe = dict(judgement="unknown")
    score = float("judgement" in mbe and mbe["judgement"] == "correct")
    return score
                

async def run_agent(
              client: ArealOpenAI,
              judge_client: ArealOpenAI,
              tokenizer: PreTrainedTokenizerFast,
              data,
              toolbox,
              processor: AutoProcessor = None,
              max_turns: int = 128,
              force_turns: int = 4,
              topk: int = 10,
              force_valid: bool = True,
              max_tokens: int = 30000,
              save_path: str | None = None,
              rank: int = -1):
    # Create client with AReaL engine and tokenizer
    # client = ArealOpenAI(engine=rollout_engine, tokenizer=tokenizer)

    # Create ASearcher Reasoning Agent
    agent = AReaLVOYAGEReasoningAgentV1(max_turns=max_turns,
                                        force_turns=force_turns,
                                        topk=topk,
                                        force_valid=force_valid)

    qid = data["id"]
    process = dict(id=data["id"],
                   question=data["question"],
                   prompt=data["question"],
                   images=data.get("images", []),
                   gt=data["answer"])
    
    completions = []
    stats = dict(
        turns=0,
        num_search=0,
        num_access=0,
        score=0.0,
    )
    cnt = 0
    while not agent.all_finished([process]):
        cnt += 1
        print(f"Agent Loop: Qid={qid} rank={rank} cnt={cnt}", flush=True)

        # Prepare query
        query = agent.prepare_queries(tokenizer, [process], processor=processor)[0]
        if query is None:
            break
        _images = query.get("images") or []
        if _images:
            _parts = [{"type": "input_text", "text": query["prompt"]}]
            for _b64 in _images:
                _url = _b64 if (isinstance(_b64, str) and _b64.startswith("data:")) else f"data:image/jpeg;base64,{_b64}"
                _parts.append({"type": "input_image", "image_url": _url})
            # Replace prompt content with structured multimodal parts (Responses-style)
            query["prompt"] = _parts
        
        
        response = None
        #
        if query["type"] == "llm":
            # Use like standard OpenAI client
            completion = await client.chat.completions.create(
                messages=[{"role": "user", "content": query["prompt"]}],
                temperature=1.0,
                max_tokens=max_tokens,
                max_completion_tokens=max(0, min(max_tokens, max_tokens - query["query_len"])),
            )
            response = completion.choices[0].message.content
            # print(f"Qid={qid} rank={rank} cnt={cnt} llm gen response: {[response]} query_len={query['query_len']} max_completion_tokens={max(0, min(max_tokens, max_tokens - query['query_len']))}")
            completions.append(completion)
            stats["turns"] += 1
        elif query["type"] == "grounding":
            # Grounding
            tool_call = f"<grounding>{query['query'][0]}</grounding>"
            response = (await toolbox.step((data["id"], [tool_call])))[0]
            stats["num_grounding"] += 1
        process = agent.consume_responses([process], [query], [response])[0]

    # Compute reward directly from predicted answer vs ground truth (MCQ A/B/C/D)
    def _extract_choice(text: Optional[str]) -> Optional[str]:
        if text is None:
            return None
        t = str(text)
        if "<answer>" in t and "</answer>" in t:
            t = t.split("<answer>")[-1].split("</answer>")[0]
        # find standalone A/B/C/D (case-insensitive)
        m = re.search(r"(?i)\b([ABCD])\b", t)
        return m.group(1).upper() if m else None

    pred_answer = agent.answers([process])[0]
    pred_choice = _extract_choice(pred_answer)

    gt = data.get("answer")
    if isinstance(gt, list):
        gt_choices = [c for c in (_extract_choice(x) for x in gt) if c]
    else:
        gt_choices = [c for c in [_extract_choice(gt)] if c]

    # Build predict sequence for compute_score
    predict_str_list = [
        h["text"] for h in process["history"] if h.get("type") == "act" and "text" in h
    ]
    # Ensure final step contains a direct answer for formatting reward
    if not any(("<answer>" in s and "</answer>" in s) for s in predict_str_list):
        predict_str_list.append(f"<think>\n\n</think>\n\n<answer>{pred_answer}</answer>")

    extra_info = {
        "acc_reward_weight": 1.0,
        "format_reward_weight": 1.0,
        "gpt_extract_answer": True,
        "extract_answer_tags": "strict",
    }
    reward_tuple = compute_score(process["question"], predict_str_list, data.get("answer"), extra_info)
    # compute_score may return a tuple (score, acc_score, format_score)
    reward = reward_tuple[0] if isinstance(reward_tuple, (list, tuple)) else reward_tuple
    stats["score"] = reward
    
    print(f"Qid={qid} rank={rank} pred_answer: {pred_answer} pred_choice: {pred_choice} gt_choices: {gt_choices} reward: {reward} stats: {stats}", flush=True)

    if save_path is not None:
        import os, json, sys
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        json.dump(process, open(save_path, "w"))

    return completions, reward, stats
