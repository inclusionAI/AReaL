import re
import time
from typing import Dict, List, Any, Optional

class AReaLSearcherQwen3Prompts:
    THINK_AND_ACT_PROMPT_v1 =  \
"""Given a question, you are an autonomous agent trying to solve the question with web browser. Given the question and the history context, generate the thought as well as the next action (only one action). Tthe completed thought should contain analysis of available information and planning for future steps. Enclose the thought within <thought> </thought> tags. 

The next action could be one of the following three, each with specific tags:
1. Search w. a search engine, e.g. <search> the search query </search>

2. Accessing some url found in prior history, e.g. <access> the url to access </access>

3. Answering the question, e.g. <answer> the answer (usually in less than 10 words) </answer> (WARNING: Answer the question only after you double check the results with sufficient search!)

Guidelines:
1. You should double check previous conclusions and identified facts using search from different perspectives. 
3. You can try different directions to solve the question, such as using different search queries.
3. If you find related entries in the search results, it is usually useful to access the corresponding urls to find more information.
4. You should find the most likely answer.
5. The next action should follow after the thought.
6. Make sure you choose only one action.
7. Carefully select the type of language to conduct your search query (Chinese or English)

Current Time: Today is 2025.07.21 

Question:
```txt
{question}
```

Reasoning history:
```txt
{history}
```

Thought: ... // the thought to be completed

Next Action: ... // the next action to be completed
"""

    THINK_AND_ACT_PROMPT = \
"""Given a question, you are an autonomous agent trying to solve the question with web browser. Given the question and the history context, generate the thought as well as the next action (only one action). The completed thought should contain a detailed analysis of current situation and a plan for future steps. The action is either a query to google search or accessing some URL. Enclose the thought within <thought> </thought> tags. 

The next action could be one of the following two, each with specific tags:
1. Search w. a search engine, e.g. <search> the search query </search>

2. Accessing some url found in prior history to find more information, e.g. <access> the url to access </access>

Guidelines:
1. You should double check previous conclusions and identified facts using search from different perspectives. 
3. You can try different directions to solve the question, such as using different search queries.
3. If you find related entries in the search results, it is usually useful to access the corresponding urls to find more information.
4. The next action should follow after the thought.
5. Make sure you should choose only one action.

Current Time: Today is 2025.07.21 

Question:
```txt
{question}
```

Reasoning history:
```txt
{history}
```

Thought: ... // the thought to be completed

Next Action: ... // the next action to be completed
"""

    THINK_AND_ANSWER_PROMPT =  \
"""Given a question, you are an autonomous agent trying to solve the question with web browser. Given the question and the history context, generate the thought as well as the final answer. The completed thought should contain detailed analysis of available information. Enclose the thought within <thought> </thought> tags, and the answer within <answer> </answer> tags.

Guideline:
1. Determine the answer based on the the available information.
2. Try to make your best guess if the found information is not enough.


Question:
```txt
{question}
```

Reasoning history:
```txt
{history}
```

Thought: ... // the thought to be completed

Final Answer: ... // the final answer
"""
    READ_PAGE_PROMPT =  \
"""Given a question, you are an autonomous agent trying to solve the question with web browser. Given the question, the history context, and the current web page, generate a thought after reading the webpage. The completed thought should contain information found related to the question, relevant links from the current webpage, and detailed analysis of available information. Enclose the thought within <thought> </thought> tags. 

Question:
```txt
{question}
```

Reasoning history:
```txt
{history}
```

Current webpage:
```txt
{content}
```

Thought: ... // the thought to be completed
"""
    READ_SEARCH_RESULTS_PROMPT =  \
"""Given a question, you are an autonomous agent trying to solve the question with web browser. Given the question, the history context, and the search results of the latest query, generate a thought after reading the search results. The completed thought should contain information found related to the question, relevant links from the latest search results that may help solve the question, and detailed analysis of available information. Enclose the thought within <thought> </thought> tags. 

Question:
```txt
{question}
```

Reasoning history:
```txt
{history}
```

Latest search results:
```txt
{content}
```

Thought: ... // the thought to be completed
"""

def process_webpage(content):
    keys = [("title", "title"), ("p", "p"), ("li", "li", lambda c: "\n" not in c)] 
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
    print(f"process the webpage: {init_length} -> {final_length}. {content[:100]}")
    return content

class AReaLSearchQwen3AgentV1:
    
    def __init__(self,
                 max_turns: int = 128,
                 force_turns: int = 1,
                 topk: int = 5,
                 force_valid: bool = False):

        self.max_turns = max_turns
        self.force_turns = force_turns
        self.force_valid = force_valid
        self.topk = topk
        # 保持与原agent相同的属性名
        self.stop = ["<|im_end|>", "<|endoftext|>"]
        self.stop_sequences = self.stop
        
        print(f"AReaLSearchAgentV1 初始化完成")

    def get_query_from_text(self, text: str) -> Optional[str]:
        pattern = r'<search>(.*?)</search>'
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return "<search>" + matches[-1].strip() + "</search>"
        
        return None
    
    def get_url_from_text(self, text: str) -> Optional[str]:
        pattern = r'<access>(.*?)</access>'
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return "<access>" + matches[-1].strip() + "</access>"
        
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

    def print_search_debug_info(self, text: str):
        query_starts = text.count('<search>')
        query_ends = text.count('</search>')
        # print(f"搜索标签统计: {query_starts}个开始标签, {query_ends}个结束标签")

    def debug_generation_tags(self, text: str) -> Dict:
        tags = {
            'query': {'open': text.count('<|begin_of_query|>'), 'close': text.count('<|end_of_query|>')},
            'documents': {'open': text.count('<|begin_of_documents|>'), 'close': text.count('<|end_of_documents|>')},
            'answer': {'open': text.count('<answer>'), 'close': text.count('</answer>')}
        }
        
        for tag_name, counts in tags.items():
            tags[tag_name]['balanced'] = counts['open'] == counts['close']
        
        return tags

    def all_finished(self, processes: List[Dict]) -> bool:
        finished = []
        for process in processes:
            finished.append(not process.get("running", True))
        return all(finished)

    def prepare_queries(self, tokenizer, processes: List[Dict]) -> List[Dict]:
        queries = []
        for process in processes:
            if "history" not in process:
                assert "pred_answer" not in process
                process["history"] = [dict(type="prompt", text=process["prompt"])]
                process["running"] = True
                process["phase"] = "search"
            
            if process["running"]:
                if "text" not in process["history"][-1] and "info_str" in process["history"][-1]:
                    # print("text not in history", process["history"][-1])
                    history = ""
                    for idx, h in enumerate(process["history"][:-1]):
                        # if idx < process.get("history_start_idx", 0):
                        #     continue
                        # do not keep intermediate reading thought
                        '''if h["type"] == "act" and idx > 0 and process["history"][idx-1]["type"] == "page" and idx + 2 < len(process["history"]) and process["history"][idx+1]["type"] == "page":
                            continue'''
                        history += h.get("short_info_str", h.get("text", ""))
                    if len(history) > 25000:
                        history = history[-25000:]
                    
                    if process["history"][-1]["type"] == "page":
                        prompt = AReaLSearcherQwen3Prompts.READ_PAGE_PROMPT.format(question=process["question"], history=history, content=process["history"][-1]["info_str"])
                    elif process["history"][-1]["type"] == "documents":
                        prompt = AReaLSearcherQwen3Prompts.READ_SEARCH_RESULTS_PROMPT.format(question=process["question"], history=history, content=process["history"][-1]["info_str"])
                    else:
                        raise RuntimeError(f"Not supported history type: {process['history'][-1]['type']}")
                    
                    input_text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False) # + process.get("cache_gen_text", "")
                    query_len = tokenizer([input_text], return_length=True)['length'][0]

                    if query_len <= 28000:
                        print(f"Reading @ Qid {process['id']}", len(tokenizer(input_text, add_special_tokens=False)["input_ids"]), len([h for h in process["history"] if h["type"] == "documents"]), len([h for h in process["history"] if h["type"] == "act"]), flush=True)
                        queries.append(dict(
                            type="llm",
                            sampling=dict(stop=self.stop, max_new_tokens=31000-query_len),
                            query_len=query_len,
                            prompt=input_text, 
                        ))
                        continue
                    
                    if "cache_gen_text" in process:
                        process.pop("cache_gen_text")
                
                if "text" in process["history"][-1]:
                    last_text = process["history"][-1]["text"]
                    if ("<search>" in last_text and 
                        last_text.strip().endswith("</search>")):
                        if True: #  len([h for h in process["history"] if h["type"] == "documents"]) < 20:
                            query_text = last_text.split("<search>")[-1].split("</search>")[0].strip()
                            if True: # not any([word in query_text.lower().split(" ") for word in ["none", "answer", "done", "exit"]]):
                                queries.append(dict(
                                    type="search", 
                                    query=[query_text.strip()], 
                                    search_params=dict(topk=self.topk)
                                ))
                                continue
                            else:
                                print("Invalid query:", query_text, flush=True)
                                process["history"].append(dict(
                                    type="documents", 
                                    info_str="\n\n<information> Invalid query </information>\n\n",
                                    short_info_str="\n\n<information> Invalid query </information>\n\n",
                                ))
                        '''else:
                            process["history"].append(dict(
                                type="documents", 
                                info_str="\n\n<information> Search Limit Reached </information>\n\n",
                                short_info_str="\n\n<information> Search Limit Reached </information>\n\n",
                            ))'''
                    elif ("<access>" in last_text and 
                        last_text.strip().endswith("</access>")):
                        query_text = last_text.split("<access>")[-1].split("</access>")[0]
                        if True: #  not any([word in query_text.lower().split(" ") for word in ["none", "answer", "done", "exit"]]):
                            queries.append(dict(
                                type="access", 
                                urls=[query_text.strip()], 
                                # search_params=dict(topk=self.topk)
                            ))
                            continue
                        else:
                            process["page_cache"] = []
                            info_str = "\n\n<information>\nNo More Information is Found for this URL.\n</information>\n\n"
                            short_info_str = "\n\n<information>\nNo More Information is Found for this URL.\n</information>\n\n"

                            process["history"].append(dict(
                                    type="page", 
                                    info_str=info_str,
                                    short_info_str=short_info_str
                                ))

                
                # input_text = "".join([h["text"] for h in process["history"]])
                history = ""
                for idx, h in enumerate(process["history"]):
                    # if idx < process.get("history_start_idx", 0):
                    #     continue
                    # do not keep intermediate reading thought
                    '''if h["type"] == "act" and idx > 0 and process["history"][idx-1]["type"] == "page" and idx + 2 < len(process["history"]) and process["history"][idx+1]["type"] == "page":
                        continue'''
                    history += h.get("short_info_str", h.get("text", ""))
                if len(history) > 25000:
                    history = history[-25000:]
                
                prompt = AReaLSearcherQwen3Prompts.THINK_AND_ACT_PROMPT.format(question=process["question"], history=history)
                input_text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False) + process.get("cache_gen_text", "")
                print(f"Generate Act for Qid {process['id']}", len(tokenizer(input_text, add_special_tokens=False)["input_ids"]), len([h for h in process["history"] if h["type"] == "documents"]), len([h for h in process["history"] if h["type"] == "act"]), flush=True)

                if any([
                    len([h for h in process["history"] if h["type"] == "documents"]) >= 20,
                    len([h for h in process["history"] if h["type"] == "act"]) >= self.force_turns,
                    process.get("phase", "search") == "answer",
                    ]):
                    process["phase"] = "answer"
                    print(f"Direct Generate Answer for Qid {process['id']}", len(tokenizer(input_text, add_special_tokens=False)["input_ids"]), len([h for h in process["history"] if h["type"] == "documents"]), len([h for h in process["history"] if h["type"] == "act"]), flush=True)
                    prompt = AReaLSearcherQwen3Prompts.THINK_AND_ACT_PROMPT_v1.format(question=process["question"], history=history)
                    input_text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False) + process.get("cache_gen_text", "")
                if self.force_valid:
                    input_text = input_text.replace('4. If you find information contradicting context of the question, you should point out that the question is invalid and the incorrect information in the question.', "4. You should find the most likely answer even when conflicting information is founded.")

                print("Query Input Length (llm):", process["id"], len(tokenizer(input_text, add_special_tokens=False)["input_ids"]),  len([h for h in process["history"] if h["type"] == "documents"]), len([h for h in process["history"] if h["type"] == "act"]), flush=True)
                if len(tokenizer(input_text, add_special_tokens=False)["input_ids"]) > 32000 or self.get_answer_from_text(process["history"][-1].get("text", "")) is not None:
                    print("process is done (1)", process["id"])
                    process["running"] = False
                    continue
                
                query_len = tokenizer([input_text], return_length=True)['length'][0]
                process["max_new_tokens"] = 31000 - query_len
                queries.append(dict(
                    type="llm", 
                    sampling=dict(stop=self.stop, max_new_tokens=process.get("max_new_tokens", 4096)),
                    query_len=query_len,
                    prompt=input_text, 
                ))
                process.pop("max_new_tokens")
        for q in queries:
            import json
            ### print([json.dumps(q, indent=4)[:1000]], flush=True)
            time.sleep(0.05)
        
        return queries

    def consume_responses(self, processes: List[Dict], queries: List[Dict], responses: List[Any]) -> List[Dict]:
        for r in responses:
            import json
            ### print([json.dumps(r, indent=4)[:1000]])
        
        i = 0
        for process in processes:
            if process["running"]:
                q, r = queries[i], responses[i]

                print("consume response", process["id"], q["type"])

                if q["type"] == "search":
                    if isinstance(r, list) and len(r) == 1:
                        r = r[0]
                    if isinstance(r, list) and isinstance(r[0], list):
                        assert all([isinstance(_r, list) and len(_r) == 1 for _r in r]), ([(type(_r) , len(_r)) for _r in r])
                        r = [_r[0] for _r in r]
                        assert all(["documents" in _r and "server_type" in _r for _r in r])
                        full_r = dict(
                            documents = [],
                            urls = [],
                            server_type = [],
                        )
                        for _r in r:
                            assert isinstance(_r["server_type"], str)
                            if "online" in _r["server_type"]:
                                _r["documents"] = ["Google Search Results: " + doc for doc in _r["documents"]]
                            full_r["documents"].extend(_r["documents"])
                            full_r["urls"].extend(_r["urls"])
                            full_r["server_type"].extend([_r["server_type"]] * len(_r["documents"]))
                        r = full_r
                    if isinstance(r, dict) and 'documents' in r:
                        documents = r["documents"]
                        urls = r["urls"]
                        server_types = r["server_type"]
                        # print(f"SearchR1RAGServer响应: {len(documents)}个文档")

                    else:
                        documents = []
                        urls = []
                        server_types = []
                    
                    print(f"搜索结果文档数量: {len(documents)}")

                    if len(documents) > 0:
                        doc_id_template = "[Doc {doc_id}]({url}):\n"
                        info_str = "\n\n<information>\n" + "\n\n".join([doc_id_template.format(doc_id=str(k+1), url=url) + doc for k, (doc, url) in enumerate(zip(documents, urls))]) + "\n</information>\n\n"
                        short_info_str = "\n\n<information>" + "\n\n".join([doc_id_template.format(doc_id=str(k+1), url=url) + doc + "..." for k, (doc, url) in enumerate(zip(documents, urls))]) + "\n</information>\n\n"

                        process["history"].append(dict(
                            type="documents", 
                            info_str=info_str,
                            short_info_str=short_info_str
                        ))
                    else:
                        process["history"].append(dict(
                            type="documents", 
                            info_str= "\n\n<information>\n" + "No Results Found." + "\n</information>\n\n",
                            short_info_str="\n\n<information>\n" + "No Results Found." + "\n</information>\n\n"
                        ))
                elif q['type'] == "access":
                    if isinstance(r, list):
                        r = r[0]
                    # process the webpage
                    # if isinstance(r, dict) and 'page' in r and isinstance(r["page"], str) and len(r["page"]) > 0:
                        # if "online" in r["server_type"] and r.get("type", "unknown") not in ["jina"]:
                        #     r["page"] = process_webpage(r["page"])
                    if isinstance(r, dict) and 'page' in r and isinstance(r["page"], str) and len(r["page"]) > 0:
                        page = r["page"]
                        page = page[:250000]
                        if "page_cache" not in process:
                            process["page_cache"] = []
                        process["page_cache"] = []
                        while len(page) > 0 and len(process["page_cache"]) < 10:
                            _len = min(10000, len(page))
                            process["page_cache"].append(f">>>> Page {len(process["page_cache"]) + 1} >>>>\n\n" + page[:_len])
                            page = page[_len:]
                        print("[DEBUG] add page", process["id"], len(r["page"]), len(process["page_cache"]), flush=True)
   
                        if "page_cache" in process and len(process["page_cache"]) > 0:
                            page = process["page_cache"].pop(0)
                            ### print(f"{process['id']} pop page cache: {[page[:100]]}")
                            info_str = "\n\n<information>" + page + "\n</information>\n\n"
                            short_info_str = "\n\n<information>\n" + page[:100] + "...\n\n" + "</information>\n\n"

                            process["history"].append(dict(
                                    type="page", 
                                    info_str=info_str,
                                    short_info_str=short_info_str
                                ))
                        
                    else:
                        page = ""
                        process["page_cache"] = []
                        info_str = "\n\n<information>\nNo More Information is Found for this URL.\n</information>\n\n"
                        short_info_str = "\n\n<information>\nNo More Information is Found for this URL.\n</information>\n\n"

                        process["history"].append(dict(
                                type="page", 
                                info_str=info_str,
                                short_info_str=short_info_str
                            ))

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
                    
                    self.print_search_debug_info(generated_text)
                    
                    extracted_thought = self.get_thought_from_text(generated_text)
                    extracted_answer = self.get_answer_from_text(generated_text)
                    extracted_query = self.get_query_from_text(generated_text)
                    extracted_url = self.get_url_from_text(generated_text)

                    # if the prompt is not asking to answer
                    if "<answer>" not in q["prompt"] and extracted_answer is not None:
                        print(f"Not time for producing answer for {process['id']}", extracted_answer, flush=True)
                        extracted_answer = None
                    
                    think_and_act = ""
                    if extracted_thought is not None:
                        think_and_act = think_and_act + extracted_thought
                    for act in [extracted_query, extracted_url, extracted_answer]:
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
                            
                        if "page_cache" in process and len(process["page_cache"]) > 0:
                            page = process["page_cache"].pop(0)
                            print(f"{process['id']} pop page cache: {[page[:100]]}")
                            info_str = "\n\n<information>" + page + "\n</information>\n\n"
                            short_info_str = "\n\n<information>\n" + page[:100] + "...\n\n" + "</information>\n\n"

                            process["history"].append(dict(
                                    type="page", 
                                    info_str=info_str,
                                    short_info_str=short_info_str
                                ))
                    elif len(raw_generated_text) == 0:
                        process["cache_gen_text"] = ""
                        process["llm_gen_fail"] = process.get("llm_gen_fail", 0) + 1
                        if process["llm_gen_fail"] > 4:
                            print("process is done (2)", process["id"], process["llm_gen_fail"])
                            process["running"] = False
                    else:
                        if process["history"][-1]["type"] in ["page", "documents"]:
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

                print("[DEBUG]  history length", process["id"], process["history"][-1]["type"], len(process["history"]), len(process.get("page_cache", [])), "page_cache" in process, len([h for h in process["history"] if h["type"] == "act"]))

                
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

