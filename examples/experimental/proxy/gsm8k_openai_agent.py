import asyncio

from agents import (
    Agent,
    ModelSettings,
    RunConfig,
    RunResult,
    SQLiteSession,
)
from agents import Runner as OpenAIRunner

from areal.api.cli_args import GenerationHyperparameters
from areal.utils import logging
from areal.utils.proxy_utils import run_and_submit_rewards

logger = logging.getLogger("OpenAIAgent")


def simplified_gsm8k_reward_fn(completions: str, answer: str):
    from areal.reward.math_parser import process_results

    return int(process_results(completions, answer)[0])


async def run_agent(messages: list[dict], run_config: RunConfig) -> RunResult:
    content = messages[-1]["content"]
    agent = Agent(name="Assistant")
    session = SQLiteSession("math")
    return await OpenAIRunner.run(
        agent, input=content, session=session, run_config=run_config
    )


async def run_agent_return_reward(data: dict) -> float:
    messages = data["messages"]
    answer = data["answer"]
    gconfig = data.get("gconfig", {})
    model_settings = GenerationHyperparameters(**gconfig).to_openai_args_dict(
        api_format="openai-agents"
    )
    run_config = RunConfig(
        model="default",  # no need to pass
        tracing_disabled=True,
        model_settings=ModelSettings(**model_settings),
    )

    result = await run_agent(messages=messages, run_config=run_config)
    reward = simplified_gsm8k_reward_fn(result.final_output, answer)
    return reward


async def run_and_submit(data: dict):
    await run_and_submit_rewards(func=run_agent_return_reward, data=data)


if __name__ == "__main__":
    import json
    import sys

    data = json.loads(sys.stdin.readline())
    asyncio.run(run_and_submit(data))
