import asyncio

from agents import Agent as OpenAIAgent
from agents import (
    ModelSettings,
    RunConfig,
    SQLiteSession,
    handoff,
)
from agents import Runner as OpenAIRunner
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX

from areal.api.cli_args import GenerationHyperparameters
from areal.utils.proxy_utils import run_and_submit_rewards


def gsm8k_reward_fn(result, answer):
    from areal.reward.math_parser import process_results

    return int(process_results(result, answer)[0])


class MultiAgentMathAgent:
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

    async def run_agent(
        self, messages: list[dict], answer: str, max_turns: int, run_config: RunConfig
    ):
        """Run the multi-agent workflow for math problem solving."""

        agent = self._create_agent_workflow()
        session = SQLiteSession("math")
        content = messages[-1]["content"]

        max_attempts = max_turns
        reward = 0

        for attempt in range(max_attempts):
            result = await OpenAIRunner.run(
                agent, input=content, session=session, run_config=run_config
            )
            reward = gsm8k_reward_fn(result=result.final_output, answer=answer)

            if reward == 1:
                break

            # If this isn't the last attempt, provide feedback for the next attempt
            if attempt < max_attempts - 1:
                content = f"""The previous attempt didn't get the correct answer.
                Please try a different approach with more careful reasoning.
                Original problem: {content}

                Previous attempt: {result.final_output}

                Please provide a new solution with step-by-step reasoning."""
            else:
                content = f"""This is your final attempt. Please be extremely careful and thorough.
                Original problem: {content}

                Previous attempts: {result.final_output}

                Please provide a final, carefully verified solution."""

        return reward


async def run_agent_return_reward(data: dict) -> float:
    messages = data["messages"]
    answer = data["answer"]
    agent_run_args = data.get("agent_run_args", {})
    gconfig = data.get("gconfig", {})

    max_turns = agent_run_args.get("max_turns", 8)
    model_settings = GenerationHyperparameters(**gconfig).to_openai_args_dict(
        api_format="openai-agents"
    )
    run_config = RunConfig(
        model="default",  # no need to pass
        tracing_disabled=True,
        model_settings=ModelSettings(**model_settings),
    )

    agent = MultiAgentMathAgent()
    reward = await agent.run_agent(
        messages=messages, answer=answer, max_turns=max_turns, run_config=run_config
    )
    return reward


async def run_and_submit(data: dict):
    await run_and_submit_rewards(func=run_agent_return_reward, data=data)


if __name__ == "__main__":
    import json
    import sys

    data = json.loads(sys.stdin.readline())
    asyncio.run(run_and_submit(data=data))
