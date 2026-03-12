"""MiniMax-based math agent implementation for AReaL training.

This module provides math agents using MiniMax's OpenAI-compatible API
that integrate with AReaL's proxy-based training infrastructure via the
AgentWorkflow pattern.

MiniMax models supported:
- MiniMax-M2.5: Peak performance, ultimate value
- MiniMax-M2.5-highspeed: Same performance, faster and more agile

API documentation: https://platform.minimax.io/docs/api-reference/text-openai-api
"""

import os

from math_verify import parse, verify
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from areal.api import AsyncRewardWrapper

MINIMAX_DEFAULT_BASE_URL = "https://api.minimax.io/v1"


def math_reward_fn(completions: str, answer: str) -> float:
    ans = parse(completions)
    gold = parse(answer)
    return float(verify(ans, gold))


class MathAgent:
    """Simple single-turn math agent using MiniMax's OpenAI-compatible API.

    MiniMax provides an OpenAI-compatible endpoint at https://api.minimax.io/v1,
    allowing seamless integration with the OpenAI SDK.

    Environment variables:
        MINIMAX_BASE_URL: API base URL (default: https://api.minimax.io/v1)
        MINIMAX_API_KEY: MiniMax API key for authentication
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs.copy()
        self.kwargs.pop("max_tokens", None)
        self.kwargs.pop("max_turns", None)
        # MiniMax temperature must be in (0.0, 1.0], default to 1.0
        self.kwargs.setdefault("temperature", 1.0)

    async def run(self, data: dict, **extra_kwargs) -> float:
        """Run the agent on a single problem.

        Args:
            data: Input data containing "messages" and "answer"
            **extra_kwargs: Contains base_url, api_key, and http_client from proxy

        Returns:
            Reward value for this trajectory
        """
        http_client = extra_kwargs.get("http_client", None)
        base_url = (
            extra_kwargs.get("base_url", None)
            or os.getenv("MINIMAX_BASE_URL")
            or MINIMAX_DEFAULT_BASE_URL
        )
        api_key = extra_kwargs.get("api_key", None) or os.getenv("MINIMAX_API_KEY")
        client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            http_client=http_client,
            max_retries=0,
        )
        comp: ChatCompletion = await client.chat.completions.create(
            messages=data["messages"], model="default", **self.kwargs
        )

        reward_fn = AsyncRewardWrapper(math_reward_fn)
        return await reward_fn(
            completions=comp.choices[0].message.content, answer=data["answer"]
        )


class MultiTurnMathAgent:
    """Multi-turn math agent using MiniMax's OpenAI-compatible API.

    Supports iterative problem solving with multiple turns, retrying
    when the answer is incorrect.

    Environment variables:
        MINIMAX_BASE_URL: API base URL (default: https://api.minimax.io/v1)
        MINIMAX_API_KEY: MiniMax API key for authentication
    """

    def __init__(self, max_turns: int = 8, **kwargs):
        self.max_turns = max_turns
        self.kwargs = kwargs.copy()
        self.kwargs.pop("max_tokens", None)
        # MiniMax temperature must be in (0.0, 1.0], default to 1.0
        self.kwargs.setdefault("temperature", 1.0)

    async def run(self, data: dict, **extra_kwargs) -> dict:
        """Run the agent on a single problem with multiple turns.

        Args:
            data: Input data containing "messages" and "answer"
            **extra_kwargs: Contains base_url, api_key, and http_client from proxy

        Returns:
            Dictionary mapping response IDs to reward values
        """
        http_client = extra_kwargs.get("http_client", None)
        base_url = (
            extra_kwargs.get("base_url", None)
            or os.getenv("MINIMAX_BASE_URL")
            or MINIMAX_DEFAULT_BASE_URL
        )
        api_key = extra_kwargs.get("api_key", None) or os.getenv("MINIMAX_API_KEY")
        messages = data["messages"].copy()
        rewards = {}
        client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            http_client=http_client,
            max_retries=0,
        )
        for _ in range(self.max_turns):
            response: ChatCompletion = await client.chat.completions.create(
                messages=messages,
                model="default",
                **self.kwargs,
            )
            message = response.choices[0].message
            messages.append(message.model_dump(exclude_none=True))
            reward_fn = AsyncRewardWrapper(math_reward_fn)
            reward = await reward_fn(
                completions=message.content, answer=data["answer"]
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
