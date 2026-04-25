# SPDX-License-Identifier: Apache-2.0

"""Math-solving agent with E2B sandbox code execution for RL training.

Uses the OpenAI Agents SDK with a ``run_python_code`` function tool.
Code is executed in an isolated E2B sandbox (KVM-level isolation),
making it safe for untrusted model-generated code.

This agent follows the AgentWorkflow pattern: it receives ``base_url``,
``api_key``, and ``http_client`` from AReaL's proxy and uses the standard
``AsyncOpenAI`` client for all LLM interactions.

Example
-------
Use with ``examples/sandbox/train_sandbox_agent.py``::

    python examples/sandbox/train_sandbox_agent.py \\
        --config examples/sandbox/gsm8k_agent_sandbox.yaml
"""

from __future__ import annotations

import os
from typing import Any

from agents import Agent, ModelSettings, OpenAIProvider, RunConfig, function_tool
from agents import Runner as OpenAIRunner
from math_verify import parse, verify
from openai import AsyncOpenAI

from areal.api import AsyncRewardWrapper
from areal.api.sandbox_api import SandboxConfig
from areal.infra.sandbox.factory import create_sandbox
from areal.utils import logging

logger = logging.getLogger("SandboxMathAgent")

# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------


def math_reward_fn(completions: str, answer: str) -> float:
    ans = parse(completions)
    gold = parse(answer)
    return float(verify(ans, gold))


# ---------------------------------------------------------------------------
# Sandbox singleton (lazily created, shared across episodes)
# ---------------------------------------------------------------------------

_sandbox = None
_sandbox_config: SandboxConfig | None = None


async def _get_sandbox(config: SandboxConfig):
    global _sandbox, _sandbox_config
    if _sandbox is None or _sandbox.is_closed:
        _sandbox_config = config
        _sandbox = await create_sandbox(config)
    return _sandbox


# ---------------------------------------------------------------------------
# Function tool
# ---------------------------------------------------------------------------


@function_tool
async def run_python_code(code: str) -> str:
    """Execute Python code in a secure sandbox and return the output.

    The sandbox has access to the Python standard library and common
    scientific packages (math, sympy, numpy, etc.). Use ``print()`` to
    produce output that will be returned.
    """
    if _sandbox is None:
        return "Error: Sandbox not initialized."
    timeout = _sandbox_config.timeout if _sandbox_config else 30.0
    result = await _sandbox.run_code(code, timeout=timeout)
    if result.success:
        output = result.text or "(no output)"
        if len(output) > 2000:
            output = output[:1000] + "\n...(truncated)...\n" + output[-500:]
        return output
    return f"Error: {result.error or result.stderr or 'Unknown error'}"


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a math problem solver. You have access to a Python code "
    "execution tool.\n"
    "When solving math problems:\n"
    "1. Think about the approach step by step.\n"
    "2. Use the run_python_code tool to perform calculations.\n"
    "3. Interpret the results and provide your final answer.\n"
    "4. Put your final numerical answer after '#### '.\n"
)


class SandboxMathAgent:
    """Agent that solves math problems using E2B sandbox code execution.

    Parameters
    ----------
    sandbox_config : dict | None
        Sandbox configuration dict (converted to ``SandboxConfig``).
        If ``None``, reads from environment variables.
    **kwargs
        Passed to ``ModelSettings`` (temperature, top_p, etc.).
    """

    def __init__(
        self,
        sandbox_config: dict | None = None,
        **kwargs: Any,
    ):
        self.sandbox_config = SandboxConfig(**(sandbox_config or {}))
        self.kwargs = kwargs.copy()
        self.kwargs.pop("max_tokens", None)
        self.kwargs.pop("max_turns", None)

    async def run(self, data: dict, **extra_kwargs: Any) -> float:
        """Run one episode: solve a math problem with sandbox tool calls."""
        http_client = extra_kwargs.get("http_client")
        base_url = extra_kwargs.get("base_url") or os.getenv("OPENAI_BASE_URL")
        api_key = extra_kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")

        client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            http_client=http_client,
            max_retries=0,
        )

        # Ensure sandbox is ready
        await _get_sandbox(self.sandbox_config)

        run_config = RunConfig(
            model_provider=OpenAIProvider(openai_client=client),
            model="default",
            tracing_disabled=True,
            model_settings=ModelSettings(**self.kwargs),
        )

        agent = Agent(
            name="Math Solver with Python Sandbox",
            instructions=SYSTEM_PROMPT,
            tools=[run_python_code],
        )

        content = data["messages"][-1]["content"]
        result = await OpenAIRunner.run(
            agent, input=content, run_config=run_config
        )

        reward_fn = AsyncRewardWrapper(math_reward_fn)
        return await reward_fn(
            completions=result.final_output, answer=data["answer"]
        )
