import asyncio

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from areal.api.cli_args import GenerationHyperparameters
from areal.reward import get_math_verify_worker
from areal.utils import logging
from areal.utils.proxy_utils import run_and_submit_rewards

logger = logging.getLogger("BasicOpenAIAgent")


def simplified_gsm8k_reward_fn(completions: str, answer: str):
    try:
        worker = get_math_verify_worker()
        return worker.verify(str(completions), str(answer))
    except Exception:
        return 0.0


class BasicOpenAIAgent:
    async def run_agent(self, messages: list[dict], **kwargs) -> str:
        async with AsyncOpenAI() as client:
            completion: ChatCompletion = await client.chat.completions.create(
                messages=messages,
                model="default",  # no need to pass
                **kwargs,
            )
            return completion.choices[0].message.content


async def run_agent_return_reward(data: dict) -> float:
    # agent_run_args = data.get("agent_run_args", {})
    # task_id = data.get("task_id", "")

    messages = data["messages"]
    answer = data["answer"]
    gconfig = data.get("gconfig", {})
    gen_args = GenerationHyperparameters(**gconfig).to_openai_args_dict(
        api_format="completions"
    )

    agent = BasicOpenAIAgent()
    content = await agent.run_agent(messages=messages, **gen_args)
    reward = simplified_gsm8k_reward_fn(content, answer)
    return reward


async def run_and_submit(data: dict):
    await run_and_submit_rewards(func=run_agent_return_reward, data=data)


# Compatible to be run in subprocess mode
if __name__ == "__main__":
    import json
    import sys

    data = json.loads(sys.stdin.readline())
    asyncio.run(run_and_submit(data))
