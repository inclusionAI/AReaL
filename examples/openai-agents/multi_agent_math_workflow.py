import asyncio
import os

from agents import Agent as OpenAIAgent
from agents import ModelSettings, OpenAIProvider, RunConfig, SQLiteSession, handoff
from agents import Runner as OpenAIRunner
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from transformers import PreTrainedTokenizerFast

from areal.api.cli_args import GenerationHyperparameters
from areal.api.reward_api import AsyncRewardWrapper
from areal.api.workflow_api import RolloutWorkflow
from areal.experimental.openai import ArealOpenAI
from areal.utils import stats_tracker


def gsm8k_reward_fn(result, answer):
    from areal.reward.math_parser import process_results

    return int(process_results(result, answer)[0])


class MultiAgentMathAgent:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        max_tokens_per_turn: int = 1024,
        max_turns: int = 8,
    ):
        self.tokenizer = tokenizer
        self.max_tokens_per_turn = max_tokens_per_turn
        self.max_turns = max_turns
        self.async_reward_fn = AsyncRewardWrapper(gsm8k_reward_fn)

    def _create_agent_workflow(self) -> OpenAIAgent:
        """Create a multi-agent workflow using handoffs for different reasoning stages."""

        # Create specialized agents for different stages
        problem_analyzer = OpenAIAgent(
            name="Problem Analyzer",
            instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
            You are a math problem analyzer. Your job is to:
            1. Carefully read and understand the math problem
            2. Identify the type of problem (algebra, geometry, arithmetic, etc.)
            3. Break down the problem into key components
            4. Identify what information is given and what needs to be found
            5. Suggest a general approach for solving the problem
            6. If you need help with the actual solution, hand off to the Solution Specialist

            Focus on understanding and analyzing the problem structure.""",
        )

        solution_specialist = OpenAIAgent(
            name="Solution Specialist",
            instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
            You are a math solution specialist. Your job is to:
            1. Take the problem analysis and create a detailed solution
            2. Show all your work and calculations step by step
            3. Use appropriate mathematical methods and formulas
            4. Provide clear explanations for each step
            5. If you need verification of your work, hand off to the Verification Agent
            6. If you need to refine your approach, hand off to the Refinement Agent

            Focus on creating accurate, well-explained solutions.""",
        )

        refinement_agent = OpenAIAgent(
            name="Refinement Agent",
            instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
            You are a refinement specialist. Your job is to:
            1. Carefully review the previous solution attempt
            2. Identify any errors, miscalculations, or areas for improvement
            3. Provide a corrected or improved solution with clear explanations
            4. Double-check all calculations and logic
            5. If you're still uncertain about the approach, hand off to the Verification Agent
            6. If the solution looks correct, hand off to the Verification Agent for final confirmation

            Focus on accuracy, thoroughness, and fixing any mistakes from the previous attempt.""",
        )

        verification_agent = OpenAIAgent(
            name="Verification Agent",
            instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
            You are a verification specialist. Your job is to:
            1. Carefully verify the solution step by step
            2. Check for any mathematical errors or logical flaws
            3. Ensure the final answer is correct and properly formatted
            4. Provide a final, verified answer with confidence
            5. If you find errors, provide the corrected solution

            This is the final stage - provide your best, most accurate answer.""",
        )

        # Create the main orchestrator agent with handoffs
        main_agent = OpenAIAgent(
            name="Math Problem Solver",
            instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
            You are a math problem solving coordinator. Your job is to:
            1. Understand the math problem presented to you
            2. Coordinate with specialized agents to solve it step by step
            3. Start by analyzing the problem structure
            4. If you need help with problem analysis, hand off to the Problem Analyzer
            5. If you need help with the solution, hand off to the Solution Specialist
            6. If the solution needs refinement, hand off to the Refinement Agent
            7. If verification is needed, hand off to the Verification Agent
            8. Ensure the final answer is correct and complete

            Use the handoff tools strategically to get the best possible solution.
            You can use multiple agents in sequence if needed for complex problems.""",
            handoffs=[
                handoff(
                    agent=problem_analyzer,
                    tool_name_override="analyze_problem",
                    tool_description_override="Analyze the problem structure and identify the approach needed",
                ),
                handoff(
                    agent=solution_specialist,
                    tool_name_override="solve_problem",
                    tool_description_override="Create a detailed solution with step-by-step work",
                ),
                handoff(
                    agent=refinement_agent,
                    tool_name_override="refine_solution",
                    tool_description_override="Refine and improve the current solution approach",
                ),
                handoff(
                    agent=verification_agent,
                    tool_name_override="verify_solution",
                    tool_description_override="Verify and finalize the solution for accuracy",
                ),
            ],
        )

        return main_agent

    async def run_agent(self, data, client: ArealOpenAI):
        """Run the multi-agent workflow for math problem solving."""
        run_config = RunConfig(
            model_provider=OpenAIProvider(openai_client=client),
            tracing_disabled=True,
            model_settings=ModelSettings(
                temperature=1.0,
                extra_args={"max_completion_tokens": self.max_tokens_per_turn},
            ),
        )

        agent = self._create_agent_workflow()
        session = SQLiteSession("math")
        content = data["messages"][-1]["content"]

        max_attempts = self.max_turns
        reward = 0

        for attempt in range(max_attempts):
            result = await OpenAIRunner.run(
                agent, input=content, session=session, run_config=run_config
            )
            reward = await self.async_reward_fn(
                result=result.final_output, answer=data["answer"]
            )

            if reward == 1:
                break

            # If this isn't the last attempt, provide feedback for the next attempt
            if attempt < max_attempts - 1:
                content = f"""The previous attempt didn't get the correct answer.
                Please try a different approach with more careful reasoning.
                Original problem: {data["messages"][-1]["content"]}

                Previous attempt: {result.final_output}

                Please provide a new solution with step-by-step reasoning."""
            else:
                content = f"""This is your final attempt. Please be extremely careful and thorough.
                Original problem: {data["messages"][-1]["content"]}

                Previous attempts: {result.final_output}

                Please provide a final, carefully verified solution."""

        client.set_final_reward(reward)

        return reward


class MultiAgentRLVRAgentWorkflow(RolloutWorkflow):
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
        # NOTE(refactor): stop tokens are not used in this workflow, adding stop and pad token ids may not be necessary
        self.gconfig = gconfig.new_with_stop_and_pad_token_ids(tokenizer)
        self.gconfig.n_samples = 1
        self.tokenizer = tokenizer
        self.dump_dir = dump_dir
        self.max_tokens = max_tokens
        self.rollout_stat_scope = rollout_stat_scope
        if self.dump_dir is not None and not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir, exist_ok=True)

        # Search hyper-parameters
        self.n_trajs = n_trajs
        self.agent = MultiAgentMathAgent(
            tokenizer=self.tokenizer,
            max_tokens_per_turn=self.gconfig.max_new_tokens,
            max_turns=max_turns,
        )

    async def arun_episode(self, engine, data):
        clients: list[ArealOpenAI] = [
            ArealOpenAI(
                engine=engine,
                tokenizer=self.tokenizer,
                tool_call_parser="qwen25",
                chat_template_type="concat",
            )
            for _ in range(self.n_trajs)
        ]

        # Collect trajectories
        rewards = await asyncio.gather(
            *[
                self.agent.run_agent(
                    data=data,
                    client=clients[i],
                )
                for i in range(self.n_trajs)
            ]
        )
        for reward in rewards:
            stats_tracker.get(self.rollout_stat_scope).scalar(reward=reward)

        interactions_with_reward = {}
        for client in clients:
            client.apply_reward_discount(turn_discount=0.9)
            interactions = client.export_interactions(style="concat")
            interactions_with_reward.update(interactions)

        return interactions_with_reward
