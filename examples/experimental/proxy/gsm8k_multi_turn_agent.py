import asyncio

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from areal.api.cli_args import GenerationHyperparameters
from areal.utils import logging
from areal.utils.proxy_utils import run_and_submit_rewards

logger = logging.getLogger("GSM8K Multi-Turn Agent")


def simplified_gsm8k_reward_fn(completions: str, answer: str):
    from areal.reward.math_parser import process_results

    return int(process_results(completions, answer)[0])


class MultiTurnMathAgent:
    async def run_agent(
        self, messages: list[dict], answer: str, max_turns: int, **kwargs
    ) -> dict[str, float]:
        messages = messages.copy()
        rewards = {}
        async with AsyncOpenAI() as client:
            for _ in range(max_turns):
                response: ChatCompletion = await client.chat.completions.create(
                    messages=messages,
                    model="default",  # no need to pass
                    **kwargs,
                )
                message = response.choices[0].message
                messages.append(message.to_dict())
                reward = simplified_gsm8k_reward_fn(
                    completions=message.content, answer=answer
                )
                rewards[response.id] = reward
                if reward == 1:
                    break
                else:
                    messages.append(
                        {
                            "role": "user",
                            "content": "Your answer is either wrong or not parsable to the reward function. You may misunderstand the original question. "
                            "Please carefully read the original question, check the previous errors, and try to answer it again.",
                        }
                    )
        return rewards


async def run_agent_return_rewards(data: dict) -> dict[str, float]:
    agent_run_args = data.get("agent_run_args", {})
    max_turns = agent_run_args.get("max_turns", 8)

    messages = data["messages"]
    answer = data["answer"]
    gconfig = data.get("gconfig", {})
    gen_args = GenerationHyperparameters(**gconfig).to_openai_args_dict(
        api_format="completions"
    )

    agent = MultiTurnMathAgent()
    rewards = await agent.run_agent(
        messages=messages, answer=answer, max_turns=max_turns, **gen_args
    )
    return rewards


async def run_and_submit(data: dict):
    await run_and_submit_rewards(func=run_agent_return_rewards, data=data)


# Compatible to be run in subprocess mode
if __name__ == "__main__":
    import json
    import sys

    data = json.loads(sys.stdin.readline())
    asyncio.run(run_and_submit(data))
