import queue
import re
from dataclasses import dataclass, asdict
import stat
import token
from typing import Dict, List, Tuple, Optional
from PIL import Image
from numpy import full

@dataclass
class Record:
    
    type: str # prompt/llm_gen/voyage_results
    text: str #text come from 1. prompt: directly from input; 2. llm_gen: llm decode; 3. voyage_results: grounding preserved text
    token_ids: List[int]
    # for prompt and voyage_results
    images: Optional[List[Image.Image]] = None
    # RL data
    input_len: Optional[int] = None
    input_tokens: Optional[List[int]] = None
    output_len: Optional[int] = None
    output_tokens: Optional[List[int]] = None
    output_logprobs: Optional[List[float]] = None
    output_versions: Optional[List[int]] = None
    status: Optional[str] = None # for voyage_results: success/error

    def to_dict(self):
        return asdict(self)

class AgentMemory:
    def __init__(self, prompt: str, prompt_token_ids: List[int], images: Optional[List[Image.Image]] = None):
        self.memory = [Record(type="prompt", text=prompt, token_ids=prompt_token_ids, images=images)]
    
    def llm_gen_count(self):
        return sum([r.type == "llm_gen" for r in self.memory])
    
    def filter_records(self, record_type):
        return [r for r in self.memory if r.type == record_type]
    
    def prepare_prompt(self):
        prompt = ""
        for r in self.memory:
            if r.type == "prompt":
                prompt = r.text
            elif r.type in ["grounding"]: #grounding is an image-based tool call, so prompt=grounding_system_prompt (we have packed text in tool)
                prompt =  prompt + "\n\n" + r.text +"\n\n"
            elif r.type == "llm_gen":
                prompt = prompt + r.text
            else:
                raise RuntimeError(f"Unknown record type: {r.type}")
        return prompt
    
    def prepare_prompt_token_ids(self):
        prompt_token_ids = []
        for r in self.memory:
            prompt_token_ids += r.token_ids
        return prompt_token_ids

    def add_record(self, r: Record):
        self.memory.append(r)
    
    def logging_stats(self) -> Dict:
        llm_gens = self.filter_records(record_type="llm_gen")
        grounding_results = self.filter_records(record_type="grounding")
        ret = dict(
            num_llm_gens=len(llm_gens),
            num_input_tokens=sum([len(r.input_tokens) for r in llm_gens]),
            num_output_tokens=sum([len(r.output_tokens) for r in llm_gens]),
            num_grounding_queries=len(grounding_results),
            num_success_grounding_queries=len([r for r in grounding_results if r.status == "success"]),
            num_error_grounding_queries=len([r for r in grounding_results if r.status == "error"])
        )
        return ret
    
    def to_dict(self):
        return [r.to_dict() for r in self.memory]

class VoyageAgent:
    def __init__(self, input_data:Dict):
        self.prompt = input_data["messages"]
        if "images" in input_data and input_data["images"] is not None:
            self.memory = AgentMemory(prompt=self.prompt, prompt_token_ids=input_data["input_ids"],images=input_data["images"])
        else:
            self.memory = AgentMemory(prompt=self.prompt, prompt_token_ids=input_data["input_ids"])
        self.summary_job_queue = queue.Queue(128)
    
    @property
    def num_turns(self):
        return self.memory.llm_gen_count()
    
    @property
    def is_finished(self):
        pattern = r'<answer>(.*?)</answer>'
        return any([len(re.findall(pattern, r.text, re.DOTALL)) > 0 for r in self.memory.filter_records("llm_gen")])
    
    def add_summary_jobs(self, summary_jobs):
        if not isinstance(summary_jobs, list):
            summary_jobs = [summary_jobs]
        for summary_job in summary_jobs:
            assert (summary_job.get("type", "unknown") in ["grounding"]), ("Unknown summary_job type: " + summary_job.get("type", "unknown"))
            self.summary_job_queue.put_nowait(summary_job)
    
    def prepare_llm_query(self):
        prompt_token_ids = self.memory.prepare_prompt_token_ids()
        sampling_params = dict(stop=["</grounding>", "</answer>"])
        if not self.summary_job_queue.empty():
            summary_job = self.summary_job_queue.get_nowait()
            if summary_job["type"] in ["grounding"]:
                input_ids=summary_job["input_ids"]
                text=summary_job["text"]
                new_record = Record(
                    type=summary_job["type"], 
                    text=text, 
                    token_ids=input_ids,
                    images=[summary_job["image"]] if summary_job.get("image", None) is not None else None,
                    status=summary_job.get("status", None),
                )
                prompt_token_ids += input_ids
                self.memory.add_record(new_record)
                # sampling_params["stop"] = ["</think>"]
        return prompt_token_ids, sampling_params
    
    def consume_llm_response(self, resp, completion_text):
        new_record = Record(
            type="llm_gen",
            text=completion_text,
            input_len=resp.input_len,
            token_ids=resp.input_tokens,
            input_tokens=resp.input_tokens,
            output_len=resp.output_len,
            output_tokens=resp.output_tokens,
            output_logprobs=resp.output_logprobs,
            output_versions=resp.output_versions            
        )
        self.memory.add_record(new_record)

        tool_calls = []
        for pattern in [r'<grounding>(.*?)</grounding>']:
            matches = re.findall(pattern, completion_text, re.DOTALL)
            if matches:
                match = matches[-1]
                tool_calls.append(str(pattern.replace('(.*?)', match)))
        return tool_calls

    def consume_tool_response(self, res, topk=5):
        '''
        res: {
            "image"
            "input_ids"
            "text"
            "type"
            "status"
            "error_info"
        }
        '''
        # process the grounding results
        if res["type"] == "grounding":
            summary_job = dict(type="grounding_results")

            if res["image"] is not None:
                summary_job["image"] = res["image"]
            summary_job["input_ids"] = res["input_ids"]
            summary_job["text"] = res["text"]
            summary_job["status"] = res["status"]
            
            self.add_summary_jobs(summary_job)

    def get_answer(self):
        text = self.memory.prepare_prompt()
        pattern = r'<answer>(.*?)</answer>'
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[-1].strip()
        return None