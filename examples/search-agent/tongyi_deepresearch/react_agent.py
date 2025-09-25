import asyncio
import os
import time
from datetime import datetime
from typing import Dict, List

import json5
from qwen_agent.agents.fncall_agent import FnCallAgent
from qwen_agent.llm.schema import Message
from qwen_agent.settings import MAX_LLM_CALL_PER_RUN
from qwen_agent.utils.utils import build_text_completion_prompt
from transformers import PreTrainedTokenizer

from areal.experimental.openai import ArealOpenAI
from areal.utils import logging

from .prompt import *

# from tool_file import *
# from tool_scholar import *
# from tool_python import *
from .tool_search import *
from .tool_visit import *

logger = logging.getLogger("Tongyi-DeepResearch react agent")

OBS_START = "<tool_response>"
OBS_END = "\n</tool_response>"

MAX_LLM_CALL_PER_RUN = int(os.getenv("MAX_LLM_CALL_PER_RUN", 100))

TOOL_CLASS = [
    # FileParser(),
    # Scholar(),
    Visit(),
    Search(),
    # PythonInterpreter(),
]
TOOL_MAP = {tool.name: tool for tool in TOOL_CLASS}

import datetime


def today_date():
    return datetime.date.today().strftime("%Y-%m-%d")


def parse_judge_result(raw_response):
    # parse results
    import ast
    import json

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


class MultiTurnReactAgent(FnCallAgent):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_tokens_per_turn: int = 1024,
        max_llm_calls_per_run: int = 100,
        max_total_tokens: int = 32768,
    ):
        self.tokenizer = tokenizer
        self.max_tokens_per_turn = max_tokens_per_turn
        self.max_llm_calls_per_run = max_llm_calls_per_run
        self.max_total_tokens = max_total_tokens
        self.max_total_tokens_before_finishing = int(max_total_tokens * 0.8)

    def count_tokens(self, messages):
        full_message = [Message(**x) for x in messages]
        full_prompt = build_text_completion_prompt(full_message, allow_special=True)
        return len(self.tokenizer.encode(full_prompt))

    async def call_server(self, client: ArealOpenAI, messages: List[Dict]) -> str:
        completion = await client.chat.completions.create(
            messages=messages,
            temperature=1.0,
            stop=["\n<tool_response>", "<tool_response>"],
            max_tokens=self.max_total_tokens,
            use_chat_template=False,
        )
        content = completion.choices[0].message.content
        assert content, "Error: LLM response is empty."
        return completion, content

    async def run_agent(
        self, data, client: ArealOpenAI, save_path: str | None = None
    ) -> List[List[Message]]:
        start_time = time.time()
        data["qid"]
        question = data["question"]
        answer = data["answer"]
        self.user_prompt = question
        system_prompt = SYSTEM_PROMPT
        cur_date = today_date()
        system_prompt = system_prompt + str(cur_date)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        stats = dict(
            turns=0,
            num_search=0,
            num_access=0,
            score=0.0,
        )
        num_llm_calls_available = self.max_llm_calls_per_run
        completions = []
        round = 0
        while num_llm_calls_available > 0:
            # Check whether time is reached
            if time.time() - start_time > 150 * 60:  # 150 minutes in seconds
                prediction = "No answer found after 2h30mins"
                termination = "No answer found after 2h30mins"
                result = {
                    "question": question,
                    "answer": answer,
                    "messages": messages,
                    "prediction": prediction,
                    "termination": termination,
                }
                return result
            round += 1
            stats["turns"] += 1
            num_llm_calls_available -= 1
            completion, content = await self.call_server(client, messages)
            completions.append(completion)
            print(f"Round {round}: {content}")
            # if "<tool_response>" in content:
            #     pos = content.find("<tool_response>")
            #     content = content[:pos]
            messages.append({"role": "assistant", "content": content})
            if "<tool_call>" in content and "</tool_call>" in content:
                tool_call = content.split("<tool_call>")[1].split("</tool_call>")[0]
                try:
                    tool_call = json5.loads(tool_call)
                    tool_name = tool_call.get("name", "")
                    tool_args = tool_call.get("arguments", {})
                    result = self.custom_call_tool(tool_name, tool_args)
                    if tool_name == "search":
                        stats["num_search"] += 1
                    elif tool_name == "visit":
                        stats["num_access"] += 1
                except:
                    result = 'Error: Tool call is not a valid JSON. Tool call must contain a valid "name" and "arguments" field.'
                result = "<tool_response>\n" + result + "\n</tool_response>"
                # print(result)
                messages.append({"role": "user", "content": result})
            if "<answer>" in content and "</answer>" in content:
                termination = "answer"
                break
            if num_llm_calls_available <= 0 and "<answer>" not in content:
                messages[-1][
                    "content"
                ] = "Sorry, the number of llm calls exceeds the limit."

            max_tokens = self.max_total_tokens_before_finishing
            token_count = self.count_tokens(messages)
            print(f"round: {round}, token count: {token_count}")

            if token_count > max_tokens:
                print(f"Token quantity exceeds the limit: {token_count} > {max_tokens}")
                messages[-1][
                    "content"
                ] = "You have now reached the maximum context length you can handle. You should stop making tool calls and, based on all the information above, think again and provide what you consider the most likely answer in the following format:<think>your final thinking</think>\n<answer>your answer</answer>"
                completion, content = await self.call_server(client, messages)
                completions.append(completion)
                messages.append({"role": "assistant", "content": content})
                if "<answer>" in content and "</answer>" in content:
                    prediction = (
                        messages[-1]["content"]
                        .split("<answer>")[1]
                        .split("</answer>")[0]
                    )
                    termination = "generate an answer as token limit reached"
                else:
                    prediction = messages[-1]["content"]
                    termination = (
                        "format error: generate an answer as token limit reached"
                    )
                result = {
                    "question": question,
                    "answer": answer,
                    "messages": messages,
                    "prediction": prediction,
                    "termination": termination,
                    "completions": completions,
                    "stats": stats,
                }
                token_count = self.count_tokens(messages)
                if token_count > self.max_total_tokens:
                    logger.warning(
                        f"Warning: total token count {token_count} exceeds the hard limit {self.max_total_tokens}."
                    )
                return result

        if "<answer>" in messages[-1]["content"]:
            prediction = (
                messages[-1]["content"].split("<answer>")[1].split("</answer>")[0]
            )
            termination = "answer"
        else:
            prediction = "No answer found."
            termination = "answer not found"
            if num_llm_calls_available == 0:
                termination = "exceed available llm calls"
        result = {
            "question": question,
            "answer": answer,
            "messages": messages,
            "prediction": prediction,
            "termination": termination,
            "completions": completions,  # final completion
            "stats": stats,
        }
        if save_path:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=4)
            print(f"Result dumped to {save_path}")
        return result

    def custom_call_tool(self, tool_name: str, tool_args: dict, **kwargs):
        if tool_name in TOOL_MAP:
            tool_args["params"] = tool_args
            if "python" in tool_name.lower():
                result = TOOL_MAP["PythonInterpreter"].call(tool_args)
            elif tool_name == "parse_file":
                params = {"files": tool_args["files"]}

                raw_result = asyncio.run(
                    TOOL_MAP[tool_name].call(
                        params, file_root_path="./eval_data/file_corpus"
                    )
                )
                result = raw_result

                if not isinstance(raw_result, str):
                    result = str(raw_result)
            else:
                raw_result = TOOL_MAP[tool_name].call(tool_args, **kwargs)
                result = raw_result
            return result

        else:
            return f"Error: Tool {tool_name} not found"

    async def calc_reward_with_llm_judge(
        self,
        result: Dict[str, str],
        judge_client: ArealOpenAI,
    ):
        # Compute reward with LLM-as-Judge
        # judge_client = ArealOpenAI(engine=rollout_engine, tokenizer=tokenizer)
        judge_prompt_template = (
            "You are an evaluation assistant. Please determine if the predicted answer is equivalent to the labeled answer.\n"
            "You should first give your rationale for the judgement, and then give your judgement result (i.e., correct or incorrect).\n\n"
            "\n"
            "question: {question}\n"
            "ground truth answers: {gt_answer}\n"
            "pred_answer: {pred_answer}\n\n"
            "Did the model give an answer **equivalent** to the labeled answer? \n\nThe output should in the following json format:\n"
            "```json\n"
            "{{\n"
            """    "rationale": "your rationale for the judgement, as a text",\n"""
            """    "judgement": "your judgement result, can only be 'correct' or 'incorrect'\n"""
            "}}\n"
            "```\n"
            "Your output:"
        )
        pred_answer = result["prediction"]
        ground_truth = result["answer"]
        if isinstance(ground_truth, list) and len(ground_truth) == 1:
            ground_truth = str(ground_truth[0])
        judge_prompt = judge_prompt_template.format(
            question=result["question"],
            gt_answer=str(ground_truth),
            pred_answer=pred_answer[:200],
        )
        judge_completion = await judge_client.chat.completions.create(
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=1.0,
            max_tokens=8192,
            max_completion_tokens=8192,
        )
        judge_response = judge_completion.choices[0].message.content
        reward = parse_judge_result(judge_response)
        return reward

    async def make_trajectory(
        self,
        data: Dict[str, str],
        client: ArealOpenAI,
        judge_client: ArealOpenAI,
        save_path: str | None = None,
    ) -> Dict:
        result = await self.run_agent(data, client, save_path=save_path)
        reward = await self.calc_reward_with_llm_judge(result, judge_client)
        completions = result["completions"]
        last_completion = completions[-1]
        client.set_reward(last_completion.id, reward)
        stats = result["stats"]
        return last_completion, stats
