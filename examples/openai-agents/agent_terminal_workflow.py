import asyncio
import logging
import os

from agents import Agent as OpenAIAgent
from agents import ModelSettings, OpenAIProvider, RunConfig, SQLiteSession
from agents import Runner as OpenAIRunner
from terminal.env import TerminalEnv
from terminal.judge_agent import JudgeAgent, judge_from_env
from terminal.prompt import SYSTEM_PROMPT
from transformers import PreTrainedTokenizerFast

from areal.api.cli_args import GenerationHyperparameters
from areal.api.workflow_api import RolloutWorkflow
from areal.experimental.openai import ArealOpenAI
from areal.utils import stats_tracker

logger = logging.getLogger(__name__)


class TerminalAgent:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        max_tokens_per_turn: int = 1024,
        max_turns: int = 8,
        max_total_tokens: int = 32768,
        dump_dir: str | None = None,
        rollout_stat_scope: str = "rollout",
    ):
        self.tokenizer = tokenizer
        self.max_tokens_per_turn = max_tokens_per_turn
        self.max_turns = max_turns
        self.max_total_tokens = max_total_tokens
        self.dump_dir = dump_dir
        self.rollout_stat_scope = rollout_stat_scope

    async def run_agent(self, data, client: ArealOpenAI, judge_agent: JudgeAgent):
        """Run the agent workflow for terminal task execution."""
        run_config = RunConfig(
            model_provider=OpenAIProvider(
                openai_client=client,
                use_responses=True,
            ),
            tracing_disabled=True,
            model_settings=ModelSettings(
                temperature=1.0,
                extra_args={"max_completion_tokens": self.max_tokens_per_turn},
                tool_choice="auto",
                store=True,
            ),
        )

        async with TerminalEnv(
            task_name=data["task_name"],
            dump_dir=self.dump_dir,
            rollout_stat_scope=self.rollout_stat_scope,
        ) as env:
            # Create agent workflow with terminal tools
            agent = OpenAIAgent(
                name="Terminal Task Agent",
                instructions=SYSTEM_PROMPT,
                tools=env.get_tools(),
            )
            session = SQLiteSession("terminal")
            content = data["instruction"]

            max_attempts = self.max_turns
            reward = 0
            judge_reward = 0
            tracker = stats_tracker.get(self.rollout_stat_scope)

            with tracker.record_timing("run_agent_total"):
                error_count = 0.0
                attempts_used = 0.0
                for attempt in range(max_attempts):
                    attempts_used = float(attempt + 1)
                    try:
                        with tracker.record_timing("openai_runner_run"):
                            result = await OpenAIRunner.run(
                                agent,
                                input=content,
                                session=session,
                                run_config=run_config,
                                max_turns=30,
                            )
                    except Exception as e:
                        logger.error(f"Error running agent: {e}")
                        error_count += 1.0
                        break

                    with tracker.record_timing("env_validate_reward"):
                        reward = env.reward()
                    if judge_agent:
                        with tracker.record_timing("judge_agent_reward"):
                            judge_reward = await judge_agent.get_reward_from_judge(
                                session=session,
                                dockerfile_contents=data["dockerfile_contents"],
                            )
                        if judge_reward >= 0 and reward < 0.99:
                            reward = reward * 0.65 + judge_reward * 0.35

                    tracker.scalar(
                        reward=reward,
                        judge_reward=judge_reward,
                        attempt_index=float(attempt),
                        input_chars=float(len(content) if content else 0.0),
                        output_chars=float(
                            len(getattr(result, "final_output", "") or "")
                        ),
                    )

                    if isinstance(reward, float) and reward >= 0.99:
                        tracker.scalar(success=1.0)
                        break

                    if attempt < max_attempts - 1:
                        content = f"""The previous attempt didn't complete the task successfully.
                        Please try a different approach.
                        Original task: {data["instruction"]}

                        Previous attempt result: {result.final_output}

                        Please analyze what went wrong and try again with a corrected approach."""
                    else:
                        content = f"""This is your final attempt. Please be extremely careful.
                        Original task: {data["instruction"]}

                        Previous attempts: {result.final_output}

                        Please provide a final, carefully executed solution."""
                    tracker.scalar(success=0.0)

            tracker.scalar(
                final_reward=reward, attempts_used=attempts_used, errors=error_count
            )

            client.set_final_reward(reward)

            return reward


class TerminalAgentWorkflow(RolloutWorkflow):
    def __init__(
        self,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
        dump_dir: str | None = None,
        rollout_stat_scope: str = "rollout",
        n_trajs: int = 1,
        max_tokens: int = 32768,
        max_turns: int = 8,
    ):
        self.gconfig = gconfig
        self.gconfig.n_samples = 1
        self.tokenizer = tokenizer
        self.dump_dir = dump_dir
        self.max_tokens = max_tokens
        self.rollout_stat_scope = rollout_stat_scope
        if self.dump_dir is not None and not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir, exist_ok=True)

        # Search hyper-parameters
        self.n_trajs = n_trajs
        self.agent = TerminalAgent(
            tokenizer=self.tokenizer,
            max_tokens_per_turn=self.gconfig.max_new_tokens,
            max_turns=max_turns,
            max_total_tokens=max_tokens,
            dump_dir=self.dump_dir,
            rollout_stat_scope=self.rollout_stat_scope,
        )
        self.judge_agent = judge_from_env()

    async def arun_episode(self, engine, data):
        clients = [
            ArealOpenAI(
                engine=engine, tokenizer=self.tokenizer, tool_call_parser="qwen25"
            )
            for _ in range(self.n_trajs)
        ]

        # Collect trajectories
        rewards = await asyncio.gather(
            *[
                self.agent.run_agent(
                    data=data,
                    client=clients[i],
                    judge_agent=self.judge_agent,
                )
                for i in range(self.n_trajs)
            ]
        )
        for reward in rewards:
            stats_tracker.get(self.rollout_stat_scope).scalar(reward=reward)

        interactions_with_reward = {}
        for client in clients:
            client.apply_reward_discount(turn_discount=0.9)
            interactions = client.export_interactions(style="individual")
            interactions_with_reward.update(interactions)
        return interactions_with_reward
