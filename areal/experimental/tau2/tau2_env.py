import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any

import aiofiles
import aiofiles.os
from agents import FunctionTool, RunContextWrapper
from datasets import Dataset
from tau2.data_model.tasks import Task
from tau2.environment.tool import Tool
from tau2.gym.gym_agent import AgentGymEnv
from tau2.registry import registry

logger = logging.getLogger(__name__)


def get_tau2_dataset(
    domains: str | list[str],
    split: str = "train",
) -> Dataset:
    """Create a HuggingFace Dataset from tau2 task IDs for one or more domains.

    Args:
        domains: The tau2 domain name(s), e.g., 'retail', 'airline', 'telecom', or a list of them
        split: Dataset split (e.g., 'train', 'test')
        type: Dataset type (e.g., 'rl', 'sft'), only 'rl' is supported for now
        tokenizer: Tokenizer (currently unused, for future compatibility)

    Returns:
        Dataset: HuggingFace Dataset containing task_id entries with domain info
    """

    if isinstance(domains, str):
        domains = domains.split(",")

    all_task_ids = []
    for domain in domains:
        splits_loader_fn = registry.get_task_splits_loader(domain)
        if splits_loader_fn is None:
            raise ValueError(f"No task splits loader found for domain {domain}")
        splits = splits_loader_fn()
        if split not in splits:
            raise ValueError(
                f"Split {split} not found in {splits}, available splits: {list(splits.keys())} for domain {domain}"
            )
        task_ids = splits[split]
        all_task_ids.extend(
            [{"task_id": task_id, "domain": domain} for task_id in task_ids]
        )
    dataset = Dataset.from_list(all_task_ids)
    return dataset


STOP_FUNCTION_NAME = "done"
TAU2_AGENT_INSTRUCTION_SOLO = f"""
You are a customer service agent that helps the user according to the <policy> provided below.
You will be provided with a ticket that contains the user's request.
You will need to plan and call the appropriate tools to solve the ticket.

You cannot communicate with the user, only make tool calls.
Stop when you consider that you have solved the ticket.
To do so, send a message containing a single tool call to the `{STOP_FUNCTION_NAME}` tool. Do not include any other tool calls in this last message.

Always follow the policy.
""".strip()

TAU2_SYSTEM_PROMPT_SOLO = """
<instructions>
{agent_instruction}
</instructions>
<policy>
{domain_policy}
</policy>
<ticket>
{ticket}
</ticket>
""".strip()

TAU2_AGENT_INSTRUCTION = """
You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy.
""".strip()

TAU2_SYSTEM_PROMPT = """
<instructions>
{agent_instruction}
</instructions>
<policy>
{domain_policy}
</policy>
""".strip()

TAU2_FORMAT_INSTRUCTION = """
First, you MUST carefully reflect on the history of interactions. Then, reason about what should be done next, which tool to call, what arguments to use. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reflexion and reasoning, you present the tool call as a valid JSON within <action> </action> tags, for example: <action>{"name": "calculate", "arguments": {"expression": "1+2"}}</action>.
""".strip()


class Tau2RLEnv:
    """
    Tau2-bench environment wrapper that follows the same design pattern as TerminalEnv.

    This class provides a unified interface for tau2-bench tasks with support for:
    - Task initialization and cleanup
    - Tool management and execution
    - Reward calculation
    - Logging and statistics tracking
    - Async operations with retry logic
    """

    def __init__(
        self,
        domain: str = "telecom",
        task_id: str = None,
        dump_dir: str | None = None,
        retry_delay: float = 1.0,
        rollout_stat_scope: str = "rollout",
        max_steps: int = 50,
        solo_mode: bool = False,
        user_llm: str | None = None,
        user_llm_args: dict | None = None,
    ):
        """
        Initialize Tau2 environment.

        Args:
            domain: The tau2 domain name (e.g., 'retail', 'airline', 'telecom')
            task_id: Specific task ID to run
            dump_dir: Directory to save logs and outputs
            retry_delay: Delay between retries in seconds
            rollout_stat_scope: Scope for statistics tracking
            max_steps: Maximum number of steps per episode
            solo_mode: Whether to use solo mode (no user interaction)
            user_llm: User LLM to use for simulation
            user_llm_args: Arguments for user LLM
        """
        self.domain = domain
        self.task_id = task_id
        self.dump_dir = dump_dir
        self.retry_delay = retry_delay
        self.rollout_stat_scope = rollout_stat_scope
        self.max_steps = max_steps
        self.solo_mode = solo_mode
        self.user_llm = user_llm or os.getenv("TAU2_USER_LLM")
        self.user_llm_args = user_llm_args

        # Internal state
        self.env: AgentGymEnv | None = None
        self.task: Task | None = None
        self.tools: list[Tool] = []
        self.tools_schema: list[dict] = []
        self.agent_policy_doc: str = ""
        self.current_step = 0
        self.episode_done = False
        self.env_id = f"{self.domain}_{self.task_id}_{uuid.uuid4()}"

        if (
            os.getenv("TAU2_USER_LLM_API_BASE") is not None
            and os.getenv("TAU2_USER_LLM_API_KEY") is not None
            and self.user_llm_args is None
        ):
            self.user_llm_args = {
                "api_base": os.getenv("TAU2_USER_LLM_API_BASE"),
                "api_key": os.getenv("TAU2_USER_LLM_API_KEY"),
            }

    def __enter__(self) -> "Tau2RLEnv":
        """Start the tau2 environment."""
        try:
            self.env = AgentGymEnv(
                domain=self.domain,
                task_id=self.task_id,
                max_steps=self.max_steps,
                solo_mode=self.solo_mode,
                user_llm=self.user_llm,
                user_llm_args=self.user_llm_args,
            )
            obs, info = self.env.reset()
            self.task = info["task"]
            self.tools = info["tools"]
            self.tools_schema = [tool.openai_schema for tool in self.tools]
            self.agent_policy_doc = info["policy"]
            self.current_step = 0
            self.episode_done = False
            self.obs = obs
            self.rewards = []
            return self
        except Exception as e:
            print(f"Failed to start tau2 environment: {e}")
            return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the tau2 environment on exit."""
        if self.env is not None:
            try:
                # Clean up environment if needed
                self.env = None
                logger.info(f"Stopped tau2 environment for task {self.task_id}")
            except Exception as e:
                logger.warning(f"Error stopping tau2 environment: {e}")
        return False  # Don't suppress exceptions

    async def __aenter__(self) -> "Tau2RLEnv":
        """Async context manager entry."""
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        return self.__exit__(exc_type, exc_val, exc_tb)

    def get_task_info(self) -> dict:
        """Get current task information.

        Returns:
            Dictionary containing task details, tools, and policy
        """
        if self.env is None:
            raise RuntimeError(
                "Environment not started. Use Tau2RLEnv as context manager first."
            )

        return {
            "task": self.task,
            "tools": self.tools,
            "tools_schema": self.tools_schema,
            "policy": self.agent_policy_doc,
            "domain": self.domain,
            "task_id": self.task_id,
            "max_steps": self.max_steps,
            "solo_mode": self.solo_mode,
        }

    def get_tools(self) -> list[FunctionTool]:
        def _clean_schema(schema: dict) -> dict:
            """Clean schema to remove incompatible properties with agents FunctionTool."""

            def _clean_dict_recursively(d: dict) -> dict:
                """Recursively clean a dictionary of problematic properties."""
                cleaned = {}
                for key, value in d.items():
                    # Skip problematic keys entirely
                    if key in [
                        "additionalProperties",
                        "$schema",
                        "title",
                        "description",
                    ]:
                        continue

                    if isinstance(value, dict):
                        # Recursively clean nested dictionaries
                        cleaned[key] = _clean_dict_recursively(value)
                    elif isinstance(value, list):
                        # Clean lists of dictionaries
                        cleaned[key] = [
                            (
                                _clean_dict_recursively(item)
                                if isinstance(item, dict)
                                else item
                            )
                            for item in value
                        ]
                    else:
                        cleaned[key] = value

                return cleaned

            # Start with recursive cleaning
            cleaned = _clean_dict_recursively(schema)

            # Additional specific fixes
            if "properties" in cleaned and isinstance(cleaned["properties"], dict):
                for prop_name, prop_schema in cleaned["properties"].items():
                    if isinstance(prop_schema, dict):
                        # Handle $ref by replacing with basic object
                        if "$ref" in prop_schema:
                            cleaned["properties"][prop_name] = {
                                "type": "object",
                                "properties": {},
                            }

                        # Ensure nested objects have proper structure
                        if (
                            prop_schema.get("type") == "object"
                            and "properties" not in prop_schema
                        ):
                            prop_schema["properties"] = {}

            # Handle allOf, anyOf, oneOf by converting to basic object
            for key in ["allOf", "anyOf", "oneOf"]:
                if key in cleaned:
                    cleaned.pop(key, None)
                    if "type" not in cleaned:
                        cleaned["type"] = "object"
                    if "properties" not in cleaned:
                        cleaned["properties"] = {}

            # Ensure basic structure exists
            if "type" not in cleaned:
                cleaned["type"] = "object"
            if "properties" not in cleaned:
                cleaned["properties"] = {}

            return cleaned

        def _create_tool_wrapper(name: str, schema: dict):
            """Create a wrapper function for the tau2 tool."""

            async def tool_wrapper(ctx: RunContextWrapper[Any], args: str) -> str:
                """Wrapper that converts agents FunctionTool call to tau2 Tool call."""
                try:
                    # Parse JSON arguments from agents FunctionTool
                    parsed_args = json.loads(args) if args else {}
                    tool_call_action = {
                        "name": name,
                        "arguments": parsed_args,
                    }
                    action_str = f"{json.dumps(tool_call_action)}"
                    obs, reward, done, _, _ = self.env.step(action_str)
                    await self._log_obs(obs, action_str, done, reward)
                    self.obs = obs
                    self.episode_done = done
                    self.rewards.append(reward)

                    return self.get_last_obs()
                except json.JSONDecodeError as e:
                    return f"Error in {name}: Invalid JSON in arguments: {e}"
                except TypeError as e:
                    return f"Error in {name}: Invalid arguments: {e}"
                except Exception as e:
                    return f"Error in {name}: {e}"

            return tool_wrapper

        function_tools = []

        for tool in self.tools:
            try:
                # Get and clean the schema
                raw_schema = tool.params.model_json_schema()
                cleaned_schema = _clean_schema(raw_schema)

                # Create the function tool
                function_tool = FunctionTool(
                    name=tool.name,
                    description=tool._get_description(),
                    params_json_schema=cleaned_schema,
                    on_invoke_tool=_create_tool_wrapper(tool.name, cleaned_schema),
                )

                function_tools.append(function_tool)

            except Exception as e:
                # Log the error but continue with other tools
                logger.error(f"Failed to create FunctionTool for {tool.name}: {e}")

        return function_tools

    def step(self, action: str) -> tuple[str, float, bool, dict]:
        """
        Execute one step in the environment.

        Args:
            action: Action string to execute

        Returns:
            Tuple of (observation, reward, done, info)
        """
        if self.env is None:
            raise RuntimeError(
                "Environment not started. Use Tau2RLEnv as context manager first."
            )

        try:
            obs, reward, done, _, _ = self.env.step(action)
            self.obs = obs if obs not in ["", None] else self.obs
            self.current_step += 1
            self.episode_done = done

            return obs, float(reward), self.episode_done, {"step": self.current_step}

        except Exception as e:
            logger.error(f"Error executing step: {e}")
            return f"Error: {e}", 0.0, True, {"error": str(e)}

    def reset(self) -> tuple[str, dict]:
        """
        Reset the environment.

        Returns:
            Tuple of (initial_observation, info)
        """
        if self.env is None:
            raise RuntimeError(
                "Environment not started. Use Tau2RLEnv as context manager first."
            )

        try:
            obs, info = self.env.reset()
            self.obs = obs  # Initialize current observation
            self.task = info["task"]
            self.tools = info["tools"]
            self.tools_schema = [tool.openai_schema for tool in self.tools]
            self.agent_policy_doc = info["policy"]
            self.current_step = 0
            self.episode_done = False

            return obs, info

        except Exception as e:
            logger.error(f"Error resetting environment: {e}")
            raise RuntimeError(f"Failed to reset environment: {e}")

    def is_done(self) -> bool:
        """Check if the current episode is done."""
        return self.episode_done

    def get_current_step(self) -> int:
        """Get the current step number."""
        return self.current_step

    def get_current_step_info(self) -> str:
        """Get information about the current step for agent decision making."""
        if self.env is None:
            return "Environment not initialized"

        info_parts = []
        info_parts.append(f"Step {self.current_step + 1}/{self.max_steps}")

        if self.obs:
            info_parts.append(f"Current observation: {self.obs}")

        if self.task:
            info_parts.append(f"Task: {self.task.ticket}")

        info_parts.append(f"Available tools: {[tool.name for tool in self.tools]}")

        return "\n".join(info_parts)

    def get_system_prompt(self) -> str:
        """Get the system prompt for the environment."""
        if self.task is not None and self.task.ticket is not None:
            # solo mode
            agent_instruction = TAU2_AGENT_INSTRUCTION_SOLO
            return TAU2_SYSTEM_PROMPT_SOLO.format(
                agent_instruction=agent_instruction,
                domain_policy=self.agent_policy_doc,
                ticket=self.task.ticket,
            )
        else:
            agent_instruction = TAU2_AGENT_INSTRUCTION
            return TAU2_SYSTEM_PROMPT.format(
                agent_instruction=agent_instruction, domain_policy=self.agent_policy_doc
            )

    def get_last_obs(self) -> str:
        """
        Get the last observation.
        Trims the leading "user: " or "tool: " if it exists.
        """
        obs = self.obs
        if obs:
            # Remove "user: " or "tool: " prefix if present
            if obs.startswith("user: "):
                return obs[len("user: ") :].strip()
            elif obs.startswith("tool: "):
                return obs[len("tool: ") :].strip()
            else:
                return obs.strip()
        return ""

    async def send_assistant_message(self, message: str) -> bool:
        """
        Send an assistant message to the environment.
        """
        obs, reward, done, _, _ = self.env.step(message)
        self.obs = obs if obs not in ["", None] else self.obs
        self.episode_done = done
        self.rewards.append(reward)
        await self._log_obs(obs, message, done, reward)
        return done

    def get_rewards(self) -> list[float]:
        """Get the list of rewards collected during the episode."""
        return self.rewards

    def get_env_id(self) -> str:
        """Get the environment ID."""
        return self.env_id

    async def _log_obs(self, obs: str, input: str, done: bool, reward: float):
        """Log observation to file asynchronously.

        Args:
            obs: Observation to log
            container_name: Name of the container for the log filename
        """
        if self.dump_dir is not None:
            dump_path = Path(self.dump_dir) / "tau2"
            await aiofiles.os.makedirs(dump_path, exist_ok=True)
            log_file = dump_path / f"{self.env_id}.jsonl"
            async with aiofiles.open(log_file, "a", encoding="utf-8") as f:
                log_entry = {
                    "obs": obs,
                    "input": input,
                    "done": done,
                    "reward": reward,
                }
                await f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
