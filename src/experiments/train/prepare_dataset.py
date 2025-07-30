import argparse
import json
from pathlib import Path
from typing import Callable, Optional

from datasets import Dataset, DatasetDict
from pydantic import Field
from rich.console import Console

from tau2.agent.base import is_valid_agent_history_message
from tau2.agent.llm_agent import LLMAgent, LLMSoloAgent
from tau2.data_model.message import (
    APICompatibleMessage,
    AssistantMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.simulation import Results, SimulationRun
from tau2.data_model.tasks import Task
from tau2.environment.environment import Environment
from tau2.environment.tool import Tool
from tau2.registry import registry
from tau2.utils.pydantic_utils import BaseModelNoExtra
from tau2.utils.utils import DATA_DIR

console = Console()

res_path = (
    DATA_DIR
    / "exp"
    / "qwen2.5-7b-retries"
    / "ollama_chat"
    / "qwen2.5:7b_telecom_no-user_gpt-4.1-2025-04-14_1trials.json"
)

save_path = DATA_DIR / "exp" / "data" / "test_sft_dataset"


class SFTDataPoint(BaseModelNoExtra):
    messages: list[dict] = Field(
        description="The messages exchanged between the user, agent and environment."
    )
    tools: list[dict] = Field(description="The tools used by the agent.")
    task_id: str = Field(description="The unique identifier for the task.")
    simulation_id: str = Field(description="The unique identifier for the simulation.")
    reward: float = Field(description="The reward received by the agent.")


class OpenAICompletionDataPoint(BaseModelNoExtra):
    messages: list[dict] = Field(
        description="The messages exchanged between the user, agent and environment."
    )
    tools: list[dict] = Field(description="The tools used by the agent.")
    parallel_tool_calls: bool = Field(
        description="Whether the tool calls are parallelized.", default=True
    )


def make_sft_dataset(
    result_path: str | Path,
    save_dir: Optional[str | Path] = None,
    success_only: bool = True,
) -> DatasetDict:
    """
    Make a SFT dataset from a results file.
    Dataset is saved to save_path if provided in HuggingFace format.

    Args:
        result_path: Path to the results file.
        save_dir: Path to save the dataset.
        success_only: If True, only include successful simulations.
    Returns:
        DatasetDict: A dataset containing the SFT data.
    """
    console.print(f"Loading results from {result_path}")
    results = Results.load(result_path)
    domain_name = results.info.environment_info.domain_name
    console.print(f"Domain: {domain_name}")
    agent_implementation = results.info.agent_info.implementation
    console.print(f"Agent implementation: {agent_implementation}")
    if "solo" in agent_implementation:
        solo_mode = True
    else:
        solo_mode = False
    console.print(f"Solo mode: {solo_mode}")
    console.print(
        f"Loading environment: Domain name: {domain_name}, Solo mode: {solo_mode}"
    )
    environment: Environment = registry.get_env_constructor(domain_name)(
        solo_mode=solo_mode
    )
    tools: list[Tool] = environment.get_tools()
    if solo_mode:
        tools += environment.get_user_tools()
    console.print(f"Number of tools: {len(tools)}")
    openai_tools = [tool.openai_schema for tool in tools] if tools else None

    agent_constructor = registry.get_agent_constructor(agent_implementation)
    console.print(f"Agent constructor: {agent_constructor}")

    is_solo_agent = issubclass(agent_constructor, LLMSoloAgent)
    if is_solo_agent != solo_mode:
        raise ValueError(
            f"Agent implementation {agent_implementation} is not compatible with solo mode {solo_mode}"
        )

    console.print(f"Building agent system prompt")

    def get_system_prompt(
        task,
    ):
        if issubclass(agent_constructor, LLMSoloAgent):
            agent: LLMSoloAgent = agent_constructor(
                tools=tools, domain_policy=environment.get_policy(), task=task
            )
        elif issubclass(agent_constructor, LLMAgent):
            agent: LLMAgent = agent_constructor(
                tools=tools, domain_policy=environment.get_policy()
            )
        else:
            raise ValueError(
                f"Agent implementation {agent_implementation} is not supported"
            )
        return agent.system_prompt

    tasks = {task.id: task for task in results.tasks}
    data_splits = registry.get_task_splits_loader(domain_name)()
    assert "train" in data_splits, "Train split is required"
    assert "test" in data_splits, (
        "Test split is required"
    )  # FIXME: this should be relaxed
    train_tasks, test_tasks = data_splits["train"], data_splits["test"]

    def get_split(task_id: str) -> str:
        if task_id in train_tasks:
            return "train"
        elif task_id in test_tasks:
            return "test"
        else:
            raise ValueError(f"Task {task_id} not found in data split")

    sft_dataset = {"train": [], "test": []}
    num_discarded = 0
    for simulation in results.simulations:
        if success_only and simulation.reward_info.reward != 1:
            num_discarded += 1
            continue
        task = tasks[simulation.task_id]
        datapoint = make_sft_from_simulation(
            task, get_system_prompt, simulation, openai_tools
        )
        sft_dataset[get_split(task.id)].append(datapoint.model_dump())
    console.print(
        f"Number trajectories: {len(sft_dataset['train'])} (train) and {len(sft_dataset['test'])} (test), {num_discarded} discarded out of {len(results.simulations)}"
    )
    sft_dataset = DatasetDict(
        {
            "train": Dataset.from_list(sft_dataset["train"]),
            "test": Dataset.from_list(sft_dataset["test"]),
        }
    )
    if isinstance(save_dir, str):
        save_dir = Path(save_dir)
    if isinstance(result_path, str):
        result_path = Path(result_path)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / f"{result_path.stem}"
    console.print(f"Saving dataset to {save_path}")
    if save_path.exists():
        console.print(f"Dataset already exists at {save_path}")
        user_input = input("Overwrite? (y/n): ")
        if user_input.lower().strip() != "y":
            console.print("Exiting...")
            return
    sft_dataset.save_to_disk(save_path)
    return sft_dataset


def to_openai_sft_dataset(
    dataset_path: str | Path,
    save_dir: str | Path,
    split: str = "train",
    success_only: bool = True,
) -> None:
    """
    Convert a dataset to an OpenAI SFT dataset in jsonl chat completion format.
    Dataset is saved to save_path if provided in HuggingFace format.

    Args:
        dataset_path: Path to the dataset.
        save_path: Path to save the dataset.
        split: Split to convert.
        success_only: If True, only include successful simulations.
    """
    if isinstance(dataset_path, str):
        dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path {dataset_path} does not exist")
    dataset = DatasetDict.load_from_disk(dataset_path)
    if split not in dataset:
        raise ValueError(f"Split {split} not found in dataset")
    datapoints = dataset[split]
    num_discarded = 0
    openai_datapoints = []
    for datapoint in datapoints:
        if success_only and datapoint["reward"] != 1:
            num_discarded += 1
            continue
        openai_datapoint = OpenAICompletionDataPoint(
            messages=datapoint["messages"],
            tools=datapoint["tools"],
        )
        openai_datapoints.append(openai_datapoint)
    console.print(
        f"Converted {len(openai_datapoints)} datapoints to OpenAI format, {num_discarded} discarded out of {len(datapoints)}"
    )
    if isinstance(save_dir, str):
        save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{split}_{dataset_path.stem}.jsonl"
    if save_path.exists():
        console.print(f"Dataset already exists at {save_path}")
        user_input = input("Overwrite? (y/n): ")
        if user_input.lower().strip() != "y":
            console.print("Exiting...")
            return
    console.print(f"Saving dataset to {save_path}")
    with open(save_path, "w") as f:
        for datapoint in openai_datapoints:
            f.write(datapoint.model_dump_json() + "\n")


def make_sft_from_simulation(
    task: Task,
    get_system_prompt: Callable,
    simulation: SimulationRun,
    openai_tools: dict,
) -> SFTDataPoint:
    """
    Make a single datapoint from a simulation.
    """
    system_prompt = get_system_prompt(task)
    assert simulation.task_id == task.id
    system_message = SystemMessage(role="system", content=system_prompt)
    messages = to_openai_train_messages(
        [system_message] + prepare_messages(simulation.messages[:])
    )
    datapoint = SFTDataPoint(
        messages=messages,
        tools=openai_tools,
        task_id=task.id,
        simulation_id=simulation.id,
        reward=simulation.reward_info.reward,
    )
    return datapoint


def prepare_messages(
    messages: list[APICompatibleMessage],
) -> list[APICompatibleMessage]:
    """
    Prepare the messages for the dataset.
    Filters out messages that are should not be part of the agent trajectory.
    """
    final_messages = [msg for msg in messages if is_valid_agent_history_message(msg)]
    return final_messages


def to_openai_train_messages(messages: list[APICompatibleMessage]) -> list[dict]:
    """
    Convert a list of Tau2 messages to a list of OpenAI train messages.
    """
    openai_train_messages = []
    for message in messages:
        if isinstance(message, UserMessage):
            openai_train_messages.append({"role": "user", "content": message.content})
        elif isinstance(message, AssistantMessage):
            tool_calls = None
            if message.is_tool_call():
                tool_calls = [
                    {
                        "name": tc.name,
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                        "type": "function",
                    }
                    for tc in message.tool_calls
                ]
            openai_train_messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": tool_calls,
                }
            )
        elif isinstance(message, ToolMessage):
            openai_train_messages.append(
                {
                    "role": "tool",
                    "content": message.content,
                }
            )
        elif isinstance(message, SystemMessage):
            openai_train_messages.append({"role": "system", "content": message.content})
    return openai_train_messages


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subparser for make_sft_dataset
    make_parser = subparsers.add_parser(
        "make", help="Create SFT dataset from Tau2 result file"
    )
    make_parser.add_argument("--result-path", type=str, required=True)
    make_parser.add_argument("--save-path", type=str, required=True)
    make_parser.add_argument("--success-only", action="store_true")

    # Subparser for to_openai_sft_dataset
    convert_parser = subparsers.add_parser(
        "to-openai", help="Convert existing SFT dataset to OpenAI format"
    )
    convert_parser.add_argument("--dataset-path", type=str, required=True)
    convert_parser.add_argument("--save-dir", type=str, required=True)
    convert_parser.add_argument("--split", type=str, default="train")
    convert_parser.add_argument("--success-only", action="store_true")

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    if args.command == "make":
        make_sft_dataset(args.result_path, args.save_path, args.success_only)
    elif args.command == "to-openai":
        to_openai_sft_dataset(
            args.dataset_path, args.save_dir, args.split, args.success_only
        )


if __name__ == "__main__":
    main()
