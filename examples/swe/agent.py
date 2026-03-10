"""SWE Agent Workflow for AReaL proxy mode.

This module implements a SWE-bench agent that uses the AReaL proxy server
for OpenAI-compatible API calls during RL training. It wraps the existing
SWEAgent from ../SWEAgent to route LLM calls through AReaL's proxy,
enabling on-policy RL training.
"""

import asyncio
import os
import sys
import time
import uuid
from datetime import datetime
from typing import Any

from areal.utils import logging

logger = logging.getLogger("SWEAgent")


def _ensure_swe_agent_importable(swe_agent_root: str = "") -> str:
    """Ensure the SWEAgent package is importable by adding its root to sys.path.

    Path resolution order:
    1. Explicit swe_agent_root argument
    2. SWE_AGENT_ROOT environment variable
    3. ../SWEAgent relative to the AReaL repository root (default layout)

    Returns the resolved SWEAgent root directory.
    """
    if swe_agent_root:
        root = os.path.abspath(swe_agent_root)
    elif os.getenv("SWE_AGENT_ROOT"):
        root = os.environ["SWE_AGENT_ROOT"]
    else:
        # Default: SWEAgent lives at ../SWEAgent relative to the AReaL root.
        # This file is at examples/swe/agent.py inside the AReaL repo.
        areal_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        root = os.path.join(os.path.dirname(areal_root), "SWEAgent")

    if os.path.isdir(root) and root not in sys.path:
        sys.path.insert(0, root)
        logger.info(f"Added SWEAgent root to sys.path: {root}")
    elif not os.path.isdir(root):
        logger.warning(
            f"SWEAgent root does not exist: {root}. "
            "Set SWE_AGENT_ROOT env var or econfig.swe_agent_root to the correct path."
        )
    return root


class SWEAgentWorkflow:
    """SWE-bench agent workflow for AReaL proxy mode.

    This workflow runs a SWE-bench agent that solves GitHub issues in
    sandboxed environments. The agent's LLM calls are routed through
    AReaL's proxy server for on-policy RL training.

    The workflow delegates to the existing SWEAgent code, passing the
    AReaL proxy URL as the model's base_url so that all inference
    requests go through the proxy (which captures token-level logprobs
    for PPO training).

    Args:
        econfig: SWE environment configuration dict.
        gen_args: Generation arguments (unused; token limits come from
            swe_agent_config YAML).
        timeout: Maximum time allowed for a single episode (default: 1800s).
    """

    def __init__(
        self,
        econfig: dict | None = None,
        gen_args: dict | None = None,
        timeout: float = 1800.0,
    ):
        if econfig is None:
            econfig = {}
        self.econfig = econfig
        self.gen_args = gen_args or {}
        self.timeout = econfig.get("timeout", timeout)

        # Ensure SWEAgent is importable at construction time so we get an early
        # warning if the path is wrong, rather than failing mid-training.
        _ensure_swe_agent_importable(econfig.get("swe_agent_root", ""))

    async def run(
        self, data: dict[str, Any], **extra_kwargs: Any
    ) -> dict[str, float] | float:
        """Run a SWE-bench agent episode.

        Args:
            data: Input data containing SWE-bench instance fields:
                - instance_id (str): SWE-bench instance ID
                - problem_statement (str): The GitHub issue description
                - eval_script (str): Shell script for reward evaluation
            **extra_kwargs: Additional kwargs injected by AReaL proxy infra:
                - base_url (str): Proxy server URL for the agent LLM.
                - http_client: Optional httpx.AsyncClient for requests.

        Returns:
            float: The reward from the episode (0.0 or 1.0).
        """
        base_url: str | None = extra_kwargs.get("base_url", None)
        if base_url is None:
            raise ValueError("base_url is required for SWEAgentWorkflow")

        econfig = self.econfig.copy()
        if "econfig" in data:
            econfig.update(data["econfig"])

        model_config = {
            "base_url": base_url,
            "api_key": "dummy",
            "model_name": "dummy",
        }
        config_name = econfig.get("swe_agent_config", "train")
        instance_id = data.get("instance_id", "unknown")

        logger.info(
            f"Starting SWE episode: instance_id={instance_id}, config={config_name}"
        )
        start_time = time.time()

        try:
            reward = await asyncio.wait_for(
                self._run_episode(data, config_name, model_config),
                timeout=self.timeout,
            )
        except TimeoutError:
            elapsed = time.time() - start_time
            logger.error(
                f"TIMEOUT: Instance {instance_id} exceeded {self.timeout}s "
                f"(elapsed: {elapsed:.1f}s). Discarding trajectory."
            )
            raise

        elapsed = time.time() - start_time
        logger.info(
            f"Finished SWE episode: instance_id={instance_id}, "
            f"reward={reward}, elapsed={elapsed:.1f}s"
        )
        return float(reward)

    async def _run_episode(
        self,
        data: dict[str, Any],
        config_name: str,
        model_config: dict[str, str],
    ) -> float:
        """Execute one SWE-bench episode and return the reward.

        This reimplements the core of run_agent_return_reward from
        SWEAgent/src/swe/run_swe_agent.py with two changes:
        1. The blocking random sleep is removed (AReaL controls concurrency
           via max_concurrent_rollouts).
        2. Errors during the episode return 0.0 instead of re-raising, so
           AReaL can still record the (failed) trajectory.
        """
        from src.swe.aenv_swe import AenvSWE
        from src.swe.agent_swe import SWEAgent

        reward = 0.0
        env = None

        instance_id = data["instance_id"]
        data_id = data.get("data_id", instance_id)
        problem_statement = data["problem_statement"]
        if isinstance(problem_statement, list):
            parts = [f"{i + 1}. issue:\n{p}" for i, p in enumerate(problem_statement)]
            problem_statement = "\n".join(parts)

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        trace_id = f"swe_{timestamp}_{data_id}_{uuid.uuid4().hex[:6]}"

        try:
            env = AenvSWE(
                data_id=data_id,
                data=data,
                faas_image=instance_id,
            )
            if not await env.check_env():
                logger.error(f"[{instance_id}] aenv init failed, returning reward=0.0")
                return 0.0

            swe_agent = SWEAgent(
                environment=env,
                config_name=config_name,
                data_id=data_id,
                trace_id=trace_id,
                model_config=model_config,
                workspace_dir_name=instance_id,
            )

            try:
                agent_result = await swe_agent.run(problem_statement=problem_statement)
            except Exception as e:
                logger.error(f"[{instance_id}] Agent run error: {e}")
                return 0.0

            exit_action = agent_result.get("exit_action", "")
            if exit_action == "[Multiple Tool Calls]":
                logger.warning(f"[{instance_id}] Multiple tool calls, reward=0")
                return 0.0
            if exit_action == "[model generate error]":
                logger.warning(f"[{instance_id}] Model generate error, reward=0")
                return 0.0

            final_answer = agent_result.get("final_answer")
            if final_answer:
                try:
                    reward = await env.get_reward()
                    logger.info(f"[{instance_id}] Reward: {reward}")
                except Exception as e:
                    logger.error(f"[{instance_id}] Reward computation error: {e}")
                    reward = 0.0
            else:
                logger.warning(f"[{instance_id}] No final answer, reward=0")

        except Exception as e:
            import traceback

            logger.error(
                f"[{instance_id}] Episode error: {e}\n{traceback.format_exc()}"
            )
            reward = 0.0
        finally:
            if env is not None:
                await env.release()

        return reward
