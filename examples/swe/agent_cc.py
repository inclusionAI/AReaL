"""CC (Claude Code) Agent Workflow for AReaL proxy mode.

This module implements a SWE-bench agent that uses Claude Code (CC) for
autonomous code editing, routed through AReaL's proxy server for RL training.

Traffic path:
  Claude Code CLI (AEnv sandbox)
    -> ANTHROPIC_BASE_URL -> golang proxy (localhost)
    -> REMOTE_PROXY_SERVICE_URL -> AReaL proxy /v1/messages
    -> Anthropic->OpenAI format translation -> SGLang inference (logprobs captured)
    -> OpenAI->Anthropic response back -> Claude Code CLI

Authentication:
  AReaL session key is set as ANTHROPIC_API_KEY -> Claude CLI sends it as
  x-api-key header -> golang proxy preserves it -> AReaL _extract_bearer_token()
  extracts it for session matching.
"""

import asyncio
import os
import sys
import time
from typing import Any

from areal.utils import logging

logger = logging.getLogger("CCAgent")


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


class CCAgentWorkflow:
    """Claude Code agent workflow for AReaL proxy mode.

    This workflow runs a Claude Code agent that solves GitHub issues in
    sandboxed environments. The agent's Anthropic API calls are routed
    through AReaL's proxy server for on-policy RL training.

    Unlike SWEAgentWorkflow which uses an OpenAI-compatible multi-step
    tool loop, CCAgentWorkflow delegates entirely to Claude Code CLI
    which autonomously explores and edits code in a single invocation.

    Args:
        econfig: CC environment configuration dict.
        gen_args: Generation arguments (unused; token limits come from
            train_cc.yaml config).
        timeout: Maximum time allowed for a single episode (default: 3600s).
    """

    def __init__(
        self,
        econfig: dict | None = None,
        gen_args: dict | None = None,
        timeout: float = 3600.0,
    ):
        if econfig is None:
            econfig = {}
        self.econfig = econfig
        self.gen_args = gen_args or {}
        self.timeout = econfig.get("timeout", timeout)

        _ensure_swe_agent_importable(econfig.get("swe_agent_root", ""))

    async def run(
        self, data: dict[str, Any], **extra_kwargs: Any
    ) -> dict[str, float] | float:
        """Run a Claude Code agent episode.

        Args:
            data: Input data containing SWE-bench instance fields:
                - instance_id (str): SWE-bench instance ID
                - problem_statement (str): The GitHub issue description
                - eval_script (str): Shell script for reward evaluation
            **extra_kwargs: Additional kwargs injected by AReaL proxy infra:
                - base_url (str): Proxy server URL (becomes REMOTE_PROXY_SERVICE_URL).
                - api_key (str): Session key (becomes ANTHROPIC_API_KEY for auth).

        Returns:
            float: The reward from the episode (0.0 or 1.0).
        """
        base_url: str | None = extra_kwargs.get("base_url", None)
        api_key: str | None = extra_kwargs.get("api_key", None)
        if base_url is None:
            raise ValueError("base_url is required for CCAgentWorkflow")

        econfig = self.econfig.copy()
        if "econfig" in data:
            econfig.update(data["econfig"])

        config_name = econfig.get("cc_agent_config", "train_cc")
        instance_id = data.get("instance_id", "unknown")

        logger.info(
            f"Starting CC episode: instance_id={instance_id}, config={config_name}"
        )
        start_time = time.time()

        try:
            reward = await asyncio.wait_for(
                self._run_episode(data, config_name, base_url, api_key or "dummy"),
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
            f"Finished CC episode: instance_id={instance_id}, "
            f"reward={reward}, elapsed={elapsed:.1f}s"
        )
        return float(reward)

    async def _run_episode(
        self,
        data: dict[str, Any],
        config_name: str,
        base_url: str,
        api_key: str,
    ) -> float:
        """Execute one CC agent episode and return the reward.

        Creates an AenvCC sandbox, injects AReaL proxy routing config,
        runs CCAgent (single Claude Code invocation), then evaluates reward.
        """
        from src.swe.aenv_cc import AenvCC
        from src.swe.agent_cc import CCAgent

        reward = 0.0
        env = None

        instance_id = data["instance_id"]
        data_id = data.get("data_id", instance_id)
        problem_statement = data["problem_statement"]
        if isinstance(problem_statement, list):
            parts = [f"{i + 1}. issue:\n{p}" for i, p in enumerate(problem_statement)]
            problem_statement = "\n".join(parts)

        try:
            env = AenvCC(
                data_id=data_id,
                data=data,
                faas_image=instance_id,
            )

            # Inject AReaL proxy routing into AenvCC config:
            # base_url -> REMOTE_PROXY_SERVICE_URL (AReaL proxy /v1/messages)
            # api_key -> ANTHROPIC_API_KEY (becomes x-api-key for AReaL auth)
            env.config.proxy_service_url = base_url
            env.config.proxy_api_key = api_key
            env.config.anthropic_api_key = api_key
            # anthropic_base_url is empty; golang proxy rewrites the target
            env.config.anthropic_base_url = ""

            if not await env.check_env():
                logger.error(f"[{instance_id}] aenv init failed, returning reward=0.0")
                return 0.0

            cc_agent = CCAgent(
                environment=env,
                config_name=config_name,
            )

            try:
                agent_result = await cc_agent.run(problem_statement=problem_statement)
            except Exception as e:
                logger.error(f"[{instance_id}] CCAgent run error: {e}")
                return 0.0

            exit_action = agent_result.get("exit_action", "")
            if exit_action == "[CC_NO_PATCH]":
                logger.warning(f"[{instance_id}] CC produced no patch, reward=0")
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
