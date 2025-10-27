from agents import Agent as OpenAIAgent
from agents import Runner as OpenAIRunner
from agents import SQLiteSession, set_default_openai_api

set_default_openai_api("chat_completions")

import json
import os

import requests

######## 数据输入
i = input()
data = json.loads(i)


######### 原有 agent 逻辑
agent = OpenAIAgent(
    name="RLVR",
)
session = SQLiteSession("math")
content = data["messages"][-1]["content"]
reward = 0.0


async def r():
    result = await OpenAIRunner.run(agent, input=content, session=session)

    return result


import asyncio

result = asyncio.run(r())


######## 奖励函数
def gsm8k_reward_fn(result, answer):
    from areal.reward.math_parser import process_results

    return int(process_results(result, answer)[0])


reward = gsm8k_reward_fn(result.final_output, data["answer"])

########## 发送 reward
base_url = os.environ.get("OPENAI_BASE_URL")
response = requests.post(f"{base_url}/final_reward", json={"final_reward": reward})
