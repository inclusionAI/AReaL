import asyncio

from agents import (
    Agent,
    ModelSettings,
    RunConfig,
    RunResult,
    SQLiteSession,
    function_tool,
    set_default_openai_api
)
from agents import Runner as OpenAIRunner

from areal.api.cli_args import GenerationHyperparameters
from areal.utils.proxy_utils import run_and_submit_rewards

set_default_openai_api("chat_completions")

@function_tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@function_tool
def subtract(a: float, b: float) -> float:
    """Subtract two numbers."""
    return a - b


@function_tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


@function_tool
def divide(a: float, b: float) -> float:
    """Divide two numbers."""
    if b == 0:
        raise ValueError("Division by zero is not allowed.")
    return a / b


@function_tool
def power(a: float, b: float) -> float:
    """Raise a to the power of b."""
    return a**b


@function_tool
def sqrt(a: float) -> float:
    """Calculate the square root of a number."""
    if a < 0:
        raise ValueError("Cannot compute square root of a negative number.")
    return a**0.5


######### run agent
async def run_agent(messages: list[dict], run_config: RunConfig) -> RunResult:
    # Create Python programming assistant Agent
    agent = Agent(
        name="RLVR Math with Calculator",
        instructions="Answer the user's math questions using the available calculator tools. Don't give the answer directly, you must use tools to do the mathematical calculation.",
        tools=[
            add,
            subtract,
            multiply,
            divide,
            power,
            sqrt,
        ],
    )

    content = messages[-1]["content"]
    session = SQLiteSession("math")
    return await OpenAIRunner.run(
        agent, input=content, session=session, run_config=run_config
    )


########## reward function
def gsm8k_reward_fn(result, answer):
    from areal.reward.math_parser import process_results

    return float(process_results(result, answer)[0])


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
    reward = gsm8k_reward_fn(result.final_output, answer)
    return reward


async def run_and_submit(data: dict):
    await run_and_submit_rewards(func=run_agent_return_reward, data=data)


# Compatible to be run in subprocess mode
if __name__ == "__main__":
    import json
    import sys

    data = json.loads(sys.stdin.readline())
    asyncio.run(run_and_submit(data))
