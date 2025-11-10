import asyncio
from typing import Any

from agents import Agent as OpenAIAgent
from agents import (
    ModelSettings,
    RunConfig,
    SQLiteSession,
    set_default_openai_api,
)
from agents import Runner as OpenAIRunner

set_default_openai_api("chat_completions")


######### agent definition
agent = OpenAIAgent(
    name="RLVR",
)


######### run agent
async def run_agent(data):
    content = data["messages"][-1]["content"]

    run_config = RunConfig(
        tracing_disabled=True,
        model_settings=ModelSettings(
            temperature=1.0,
            top_p=1.0,
            max_tokens=8192,
        ),
    )
    session = SQLiteSession("math")

    result = await OpenAIRunner.run(
        agent, input=content, session=session, run_config=run_config
    )

    return result


########## reward function
def gsm8k_reward_fn(result, answer):
    from areal.reward.math_parser import process_results

    return float(process_results(result, answer)[0])


async def run_agent_return_reward(data: Any) -> float:
    result = await run_agent(data)
    reward = gsm8k_reward_fn(result.final_output, data["answer"])
    return reward


if __name__ == "__main__":
    asyncio.run(
        run_agent_return_reward(
            {"messages": [{"role": "user", "content": "What is 2+2?"}], "answer": "4"}
        )
    )
