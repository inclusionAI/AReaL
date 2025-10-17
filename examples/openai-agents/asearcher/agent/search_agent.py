import queue
import re
from dataclasses import asdict, dataclass
from typing import Dict


@dataclass
class Record:
    type: str  # prompt/llm_gen/search_results/webpage
    text: str
    # token_ids: List[int]
    # for webpage and search results
    short_text: str = ""
    # RL data
    # input_len: Optional[int] = None
    # input_tokens: Optional[List[int]] = None
    # output_len: Optional[int] = None
    # full_token_ids: Optional[List[int]] = None
    # output_tokens: Optional[List[int]] = None
    # output_logprobs: Optional[List[float]] = None
    # output_versions: Optional[List[int]] = None

    def to_dict(self):
        return asdict(self)


class AgentMemory:
    def __init__(self, prompt):
        self.memory = [Record(type="prompt", text=prompt)]

    def llm_gen_count(self):
        return sum([r.type == "llm_gen" for r in self.memory])

    def filter_records(self, record_type):
        return [r for r in self.memory if r.type == record_type]

    def prepare_prompt(self):
        prompt = ""
        for r in self.memory:
            if r.type == "prompt":
                prompt = r.text
            elif r.type in ["search_results", "webpage"]:
                prompt = prompt + "\n\n" + r.short_text + "\n<think>\n"
            elif r.type == "llm_gen":
                prompt = prompt + r.text
            else:
                raise RuntimeError(f"Unknown record type: {r.type}")
        return prompt

    # def prepare_prompt_token_ids(self):
    #     prompt_token_ids = []
    #     for r in self.memory:
    #         prompt_token_ids += r.token_ids
    #     return prompt_token_ids

    def add_record(self, r: Record):
        self.memory.append(r)

    def logging_stats(self) -> Dict:
        llm_gens = self.filter_records(record_type="llm_gen")
        search_results = self.filter_records(record_type="search_results")
        webpages = self.filter_records(record_type="webpage")
        ret = dict(
            num_llm_gens=len(llm_gens),
            # num_input_tokens=sum([len(r.input_tokens) for r in llm_gens]),
            # num_output_tokens=sum([len(r.output_tokens) for r in llm_gens]),
            num_search_queries=len(search_results),
            num_success_search_queries=len(
                [
                    r
                    for r in search_results
                    if "No search results are found" not in r.text
                ]
            ),
            num_failed_search_queries=len(
                [r for r in search_results if "No search results are found" in r.text]
            ),
            num_pages=len(webpages),
            num_success_url_accesses=len(
                [r for r in webpages if ">>>> Page 1 >>>>" in r.text]
            ),
            num_failed_url_accesses=len(
                [r for r in webpages if ">>>> Page 1 >>>>" in r.text]
            ),
        )
        return ret

    def to_dict(self):
        return [r.to_dict() for r in self.memory]


class SearchAgent:
    def __init__(self, prompt):
        self.prompt = prompt
        self.memory = AgentMemory(prompt=prompt)
        self.summary_job_queue = queue.Queue(128)

    @property
    def num_turns(self):
        return self.memory.llm_gen_count()

    @property
    def is_finished(self):
        pattern = r"<answer>(.*?)</answer>"
        return any(
            [
                len(re.findall(pattern, r.text, re.DOTALL)) > 0
                for r in self.memory.filter_records("llm_gen")
            ]
        )

    def add_summary_jobs(self, summary_jobs):
        if not isinstance(summary_jobs, list):
            summary_jobs = [summary_jobs]
        for summary_job in summary_jobs:
            assert summary_job.get("type", "unkown") in [
                "search_results",
                "webpage",
            ], "Unknown summary_job type: " + summary_job.get("type", "unknown")
            self.summary_job_queue.put_nowait(summary_job)

    def prepare_llm_query(self):
        # prompt_token_ids = self.memory.prepare_prompt_token_ids()
        sampling_params = dict(stop=["</search>", "</access>", "</answer>"])
        if not self.summary_job_queue.empty():
            summary_job = self.summary_job_queue.get_nowait()
            if summary_job["type"] in ["search_results", "webpage"]:
                full_text = "\n\n" + summary_job["text"] + "\n<think>\n"
                short_text = (
                    "\n\n"
                    + summary_job.get("short_text", summary_job["text"])
                    + "\n<think>\n"
                )
                # full_token_ids, short_token_ids = tokenizer([full_text, short_text], add_special_tokens=False)["input_ids"]
                new_record = Record(
                    type=summary_job["type"],
                    text=full_text,
                    short_text=short_text,
                    # token_ids=short_token_ids,
                    # full_token_ids=full_token_ids,
                )
                # prompt_token_ids += full_token_ids
                self.memory.add_record(new_record)
                sampling_params["stop"] = ["</think>"]
        return sampling_params
        # return prompt_token_ids, sampling_params

    def consume_llm_response(self, completion_text):
        new_record = Record(
            type="llm_gen",
            text=completion_text,
            # token_ids=resp.output_tokens,
            # input_len=resp.input_len,
            # input_tokens=resp.input_tokens,
            # output_len=resp.output_len,
            # output_tokens=resp.output_tokens,
            # output_logprobs=resp.output_logprobs,
            # output_versions=resp.output_versions
        )
        self.memory.add_record(new_record)

        tool_calls = []
        for pattern in [
            r"<search>(.*?)</search>",
            r"<access>(.*?)</access>",
            r"<answer>(.*?)</answer>",
        ]:
            matches = re.findall(pattern, completion_text, re.DOTALL)
            if matches:
                match = matches[-1]
                tool_calls.append(str(pattern.replace("(.*?)", match)))

        return tool_calls

    def consume_tool_response(self, res, topk=5):
        # process the search results
        if res["type"] == "search":
            summary_job = dict(type="search_results")

            documents = res["documents"][:topk]
            urls = res["urls"][:topk]

            if len(documents) > 0:
                doc_id_template = "[Doc {doc_id}]({url}):\n"
                text = (
                    "<information>\n"
                    + "\n\n".join(
                        [
                            doc_id_template.format(doc_id=str(k + 1), url=url)
                            + doc[:5000]
                            for k, (doc, url) in enumerate(zip(documents, urls))
                        ]
                    )
                    + "\n</information>"
                )
            else:
                text = "<information>\nNo search results are found.\n</information>"

            summary_job["text"] = text
            self.add_summary_jobs(summary_job)

        # process the webpage
        elif res["type"] == "access":
            summary_jobs = []
            page = res["page"]
            if page is not None and page.strip() != "":
                page = page[:250000]
                while len(page) > 0 and len(summary_jobs) < 10:
                    _len = min(25000, len(page))
                    summary_jobs.append(
                        dict(
                            type="webpage",
                            text=f"<information>\n>>>> Page {len(summary_jobs) + 1} >>>>\n\n"
                            + page[:_len]
                            + "\n</information>",
                            short_text=f"<information>\n>>>> Page {len(summary_jobs) + 1} >>>>\n\n"
                            + page[:100]
                            + "\n</information>",
                        )
                    )
                    page = page[_len:]
            else:
                summary_jobs.append(
                    dict(
                        type="webpage",
                        text="<information>\nNo More Information is Found for this URL.\n</information>",
                    )
                )
            self.add_summary_jobs(summary_jobs)

    def get_answer(self):
        text = self.memory.prepare_prompt()
        pattern = r"<answer>(.*?)</answer>"
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[-1].strip()
        return None
