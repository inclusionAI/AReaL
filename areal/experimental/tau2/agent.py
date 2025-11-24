import os

from agents import Agent as OpenAIAgent
from agents import ModelSettings, RunConfig, SQLiteSession, set_default_openai_api
from agents import Runner as OpenAIRunner

from areal.experimental.tau2.tau2_env import Tau2RLEnv
from areal.utils import stats_tracker

set_default_openai_api("chat_completions")


async def run_agent_return_reward(data) -> float:
    async with Tau2RLEnv(
        domain=data["domain"],
        task_id=data["task_id"],
    ) as env:
        agent = OpenAIAgent(
            name="Tau2 Agent",
            instructions=env.get_system_prompt(),
            tools=env.get_tools(),
        )
        session = SQLiteSession(env.get_env_id())

        ## TAU2_MAX_STEP 包含了 user/assistant/tools总共的步数, 通常会比较大
        max_attempts = os.getenv("TAU2_MAX_STEP", 30)
        tracker = stats_tracker.get("agent")

        with tracker.record_timing("task_total"):
            error_count = 0.0
            attempts_turn = 0
            while not env.is_done() and attempts_turn < max_attempts:
                attempts_turn += 1
                try:
                    with tracker.record_timing("task_turn"):
                        result = await OpenAIRunner.run(
                            agent,
                            input=env.get_last_obs(),
                            session=session,
                            run_config=RunConfig(
                                tracing_disabled=True,
                                model_settings=ModelSettings(
                                    temperature=1.0,
                                    top_p=1.0,
                                    # NOTE: setting max_completion_tokens instead of max_tokens
                                    extra_args={"max_completion_tokens": 8192},
                                ),
                            ),
                            max_turns=10,
                        )
                        if await env.send_assistant_message(result.final_output):
                            break
                except Exception:
                    error_count += 1.0

        reward = sum(env.get_rewards())

        tracker.scalar(
            final_reward=reward,
            attempts_turn=attempts_turn,
            run_agent_error=error_count,
        )

        return reward
