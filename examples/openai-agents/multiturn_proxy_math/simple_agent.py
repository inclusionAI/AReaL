from agents import Agent, set_default_openai_api

set_default_openai_api("chat_completions")

import os
from typing import Any

import requests
from agents import FunctionTool, RunContextWrapper
from pydantic import BaseModel


def hello_user(user: str) -> str:
    return f"Hello, {user}!"


class FunctionArgs(BaseModel):
    username: str


async def run_function(ctx: RunContextWrapper[Any], args: str) -> str:
    parsed = FunctionArgs.model_validate_json(args)
    return hello_user(user=parsed.username)


tool = FunctionTool(
    name="hello_user",
    description="says hello to the user",
    params_json_schema=FunctionArgs.model_json_schema(),
    on_invoke_tool=run_function,
)


agent = Agent(
    name="Customer Entertainment",
    instructions="You are a entertainment and need to say hello to every user",
    tools=[tool],
)

from agents import Runner


async def main():
    result = await Runner.run(agent, "Bob is comming")
    print(result.final_output)


import asyncio

asyncio.run(main())


# give reward
base_url = os.environ.get("OPENAI_BASE_URL")
reward = 1.0
response = requests.post(f"{base_url}/final_reward", json={"final_reward": reward})
