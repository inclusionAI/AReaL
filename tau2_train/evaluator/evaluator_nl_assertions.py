import json

# from tau2_train.config import DEFAULT_LLM_NL_ASSERTIONS, DEFAULT_LLM_NL_ASSERTIONS_ARGS
from tau2_train.data_model.message import Message, SystemMessage, UserMessage
from tau2_train.data_model.simulation import NLAssertionCheck, RewardInfo
from tau2_train.data_model.tasks import RewardType, Task
from tau2_train.utils.llm_utils import to_litellm_messages
from tau2_train.user_simulator import create_with_retry
from openai import OpenAI
from loguru import logger
import re

def extract_json(text, min_length=5):
    code_pattern = r"(?i)```(?:json|JSON)?\s*\n?(.*?)\n?```"
    code_blocks = re.findall(code_pattern, text, re.DOTALL)
    valid_blocks = []
    for block in code_blocks:
        clean_block = block.strip()
        if len(clean_block) < min_length:
            continue

        valid_blocks.append(clean_block)

    if not valid_blocks:
        # logger.warning(f"failed to extract python code from {text}")
        return None
    return valid_blocks[-1]

class NLAssertionsEvaluator:
    """
    Judge that evaluates whether a trajectory adheres to all the natural-language assertions.
    """

    @classmethod
    def calculate_reward(
        cls,
        task: Task,
        full_trajectory: list[Message],
        llm_name: str,
        api_key: str,
        base_url: str,
    ) -> RewardInfo:
        """
        Calculate the reward for the simulation by using an LLM to evaluate whether the trajectory adheres to all the natural-language assertions
        """
        if task.evaluation_criteria is None:
            return RewardInfo(
                reward=1.0,
                nl_assertions=[],
                info={"note": "No evaluation criteria"},
                reward_breakdown={RewardType.NL_ASSERTION: 1.0},
            )
        nl_assertions = task.evaluation_criteria.nl_assertions
        if not nl_assertions:
            return RewardInfo(
                reward=1.0,
                nl_assertions=[],
                info={"note": "No nl_assertions to evaluate"},
                reward_breakdown={RewardType.NL_ASSERTION: 1.0},
            )

        nl_assertions_checks = cls.evaluate_nl_assertions(
            full_trajectory, nl_assertions,
            llm_name, api_key, base_url,
        )

        # Calculate reward: 1 if all expectations are met, 0 otherwise
        all_expectations_met = all(result.met for result in nl_assertions_checks)
        reward = 1.0 if all_expectations_met else 0.0

        return RewardInfo(
            reward=reward,
            nl_assertions=nl_assertions_checks,
            reward_breakdown={RewardType.NL_ASSERTION: reward},
        )

    @classmethod
    def evaluate_nl_assertions(
        cls,
        trajectory: list[Message],
        nl_assertions: list[str],
        llm_name: str,
        api_key: str,
        base_url: str
    ) -> list[NLAssertionCheck]:
        """
        Evaluate whether the trajectory meets each expected outcome.

        Args:
            trajectory: List of messages from the conversation
            nl_assertions: List of natural-language assertions to evaluate

        Returns:
            List of evaluation results for each NL assertion, containing:
            - nl_assertion: The NL assertion being evaluated
            - metExpectation: Boolean indicating if the assertion was met
            - reasoning: Explanation for the evaluation
        """
        trajectory_str = "\n".join(
            [f"{message.role}: {message.content}" for message in trajectory]
        )
        # System prompt similar to the TypeScript implementation
        system_prompt = """
        TASK
        - You will be given a list of expected outcomes and a conversation that was collected during a test case run.
        - The conversation is between an agent and a customer.
        - Your job is to evaluate whether the agent satisfies each of the expected outcomes.
        - Grade each expected outcome individually.

        FORMAT
        - Your response should be a JSON object with the following fields:
        - `reasoning`: a short explanation for your classification
        - `metExpectation`: `true` if the agent satisfies the expected outcomes, `false` otherwise
        - `expectedOutcome`: repeat the expectation from the input that you are grading
        
        Example response structure:
        {
            "results": [
                {
                    "expectedOutcome": "<one of the expected outcomes from the input>",
                    "reasoning": "<reasoning trace>",
                    "metExpectation": <false or true>,
                }
            ]
        }
        """

        user_prompt = f"""
        conversation:
        {trajectory_str}
        
        expectedOutcomes:
        {nl_assertions}
        """

        messages = [
            SystemMessage(role="system", content=system_prompt),
            UserMessage(role="user", content=user_prompt),
        ]
        
        llm_client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

        messages = to_litellm_messages(messages)

        try:
            response = create_with_retry(
                llm_client,
                model=llm_name,
                messages=messages,
                is_eval=True,
            )
        except:
            return []

        ## TODO
        try:
            if "```" in response.choices[0].message.content:
                logger.info(response.choices[0].message.content)
                content = extract_json(response.choices[0].message.content)
            else:
                content = response.choices[0].message.content

            result_data = json.loads(content)
            return [
                NLAssertionCheck(
                    nl_assertion=result["expectedOutcome"],
                    met=result["metExpectation"],
                    justification=result["reasoning"],
                )
                for result in result_data.get("results", [])
            ]
        except:
            logger.info(response.choices[0].message.content)
            return []