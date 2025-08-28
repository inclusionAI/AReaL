import argparse
import json
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional, TypeVar, Union

from pydantic import Field
from rich.console import Console
from datasets import Dataset

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

console = Console()


TListOrMapping = TypeVar("TListOrMapping", list, Mapping)


def remove_none_values(example: TListOrMapping) -> TListOrMapping:
    """
    # From TRL: https://github.com/huggingface/trl/blob/30576d2ddcf2c0e17c399399e2465fbe81446ade/trl/trainer/sft_trainer.py#L71
    # Use to remove None values added by Arrow/Parquet when there are mismatched keys in nested structures.
    Recursively removes entries with `None` values from a nested structure (list or dictionary).

    Args:
        example (`list` or `Mapping`):
            Input nested structure (list or dictionary) from which to remove `None`.

    Example:
    ```python
    >>> [{
    ...     "a": {"aa": None,
    ...           "ab": 1},
    ...     "b": "my_string",
    ... }]
    >>> remove_none_values(example)
    [{'a': {'ab': 1}, 'b': 'my_string'}]
    ```
    """
    if isinstance(example, list):
        return [
            remove_none_values(value) if isinstance(value, (dict, list)) else value
            for value in example
        ]
    elif isinstance(example, Mapping):
        return {
            key: remove_none_values(value) if isinstance(value, (dict, list)) else value
            for key, value in example.items()
            if value is not None
        }
    else:
        raise TypeError("Input must be a list or a dictionary.")


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


class SFTDataset(BaseModelNoExtra):
    splits: dict[str, list[SFTDataPoint]] = Field(
        description="The splits of the dataset."
    )

    def __getitem__(self, key: str) -> list[SFTDataPoint]:
        return self.splits[key]

    def __setitem__(self, key: str, value: list[SFTDataPoint]):
        self.splits[key] = value

    def __len__(self) -> int:
        return sum(len(split) for split in self.splits.values())

    def __iter__(self):
        return iter(self.splits)

    def __contains__(self, key: str) -> bool:
        return key in self.splits

    def json_dump(self, path: str | Path):
        console.print(f"Saving dataset to {path}")
        with open(path, "w") as f:
            json.dump(self.model_dump(), f, indent=4)

    @classmethod
    def json_load(cls, path: str | Path):
        console.print(f"Loading dataset from {path}")
        with open(path, "r") as f:
            return cls.model_validate_json(f.read())

    def get_info(self) -> dict:
        return {split: len(self[split]) for split in self.splits.keys()}



def load_as_hf_dataset(dataset_path, max_datapoints=None, save_to_path=None) -> Dataset:
    """
    Loads a dataset from a JSONL file of OpenAICompletionDataPoint objects and returns a HuggingFace Dataset.
    If save_to_path is provided, the dataset is saved to a HuggingFace Dataset.
    Args:
        dataset_path: Path to the dataset.
        max_datapoints: Maximum number of datapoints to load.
        save_to_path: Path to save the dataset.
    Returns:
        Dataset: A HuggingFace Dataset.
    """
    dataset = []
    with open(dataset_path, "r") as f:
        for line in f:
            datapoint = OpenAICompletionDataPoint.model_validate_json(line)
            dataset.append(datapoint)
    if max_datapoints is not None:
        dataset = dataset[:max_datapoints]
    dataset = Dataset.from_list([dp.model_dump() for dp in dataset])
    if save_to_path is not None:
        dataset.save_to_disk(save_to_path)
    return dataset


def make_sft_dataset(
    result_paths_or_dir: Union[Iterable[str | Path], str | Path],
    save_dir: Optional[str | Path] = None,
    name: Optional[str] = None,
    success_only: bool = True,
    splits: tuple[str] = ("train", "test"),
) -> SFTDataset:
    """
    Make a SFT dataset from a list of result paths.
    Dataset is saved to save_path if provided in json format.

    Args:
        result_paths_or_dir: List of result paths or a directory path containing result files.
        save_dir: Path to save the dataset.
        name: Name of the dataset (you have to provide it if save_dir is provided)
        success_only: If True, only include successful simulations.
        splits: Splits to include in the dataset. Default is ("train", "test").
    """

    if isinstance(result_paths_or_dir, str):
        result_paths_or_dir = Path(result_paths_or_dir)
    if isinstance(result_paths_or_dir, Path):
        result_paths = list(result_paths_or_dir.glob("*.json"))
    elif isinstance(result_paths_or_dir, Iterable):
        result_paths = list(result_paths_or_dir)
    else:
        raise ValueError(
            f"Invalid type for result_paths_or_dir: {type(result_paths_or_dir)}"
        )

    if save_dir is not None and name is None:
        raise ValueError("Name is required if save_dir is provided")
    if isinstance(save_dir, str):
        save_dir = Path(save_dir)
    if save_dir is not None:
        save_path = save_dir / f"{name}.json"
    else:
        save_path = None
    splits_data = {split: [] for split in splits}
    for result_path in result_paths:
        dataset = make_sft_dataset_from_results(
            result_path=result_path, success_only=success_only
        )
        for split in splits:
            if split not in dataset:
                raise ValueError(f"Dataset from {result_path} is missing split {split}")
            splits_data[split] += dataset[split]

    sft_dataset = SFTDataset(splits=splits_data)
    console.print_json(json.dumps(sft_dataset.get_info(), indent=2))
    do_save = save_path is not None
    if do_save and not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    if save_path.exists() and do_save:
        console.print(f"Dataset already exists at {save_path}")
        user_input = input("Overwrite? (y/n): ")
        do_save = user_input.lower().strip() == "y"
    if do_save:
        sft_dataset.json_dump(save_path)
    else:
        console.print("Exiting...")
    return sft_dataset


def make_sft_dataset_from_results(
    result_path: str | Path,
    save_dir: Optional[str | Path] = None,
    name: Optional[str] = None,
    success_only: bool = True,
) -> list[SFTDataPoint]:
    """
    Make a SFT dataset from a results file.
    Dataset is saved to save_path if provided in json format.

    Args:
        result_path: Path to the results file.
        save_dir: Path to save the dataset.
        name: Name of the dataset (if None, it will be the name of the results file)
        success_only: If True, only include successful simulations.
    Returns:
        list[SFTDataPoint]: A list of SFT data points.
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
    sft_dataset = SFTDataset(splits=sft_dataset)
    console.print_json(json.dumps(sft_dataset.get_info(), indent=2))
    # save dataset if save_dir is provided
    do_save = save_dir is not None
    if not do_save:
        return sft_dataset

    if isinstance(save_dir, str):
        save_dir = Path(save_dir)
    if isinstance(result_path, str):
        result_path = Path(result_path)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)

    if name is None:
        name = result_path.stem
    save_path = save_dir / f"{name}.json"

    if save_path.exists() and do_save:
        console.print(f"Dataset already exists at {save_path}")
        user_input = input("Overwrite? (y/n): ")
        do_save = user_input.lower().strip() == "y"
    if do_save:
        sft_dataset.json_dump(save_path)
    else:
        console.print("Exiting...")
    return sft_dataset


def to_openai_sft_dataset(
    dataset_path: str | Path,
    save_dir: str | Path,
    split: str = "train",
    success_only: bool = True,
) -> list[OpenAICompletionDataPoint]:
    """
    Convert a dataset to an OpenAI SFT dataset in jsonl chat completion format.
    Removes messages after the last agent message.
    Dataset is saved to save_path if provided in HuggingFace format.

    Args:
        dataset_path: Path to the dataset.
        save_path: Path to save the dataset.
        split: Split to convert.
        success_only: If True, only include successful simulations.
    """

    def remove_messages_after_last_agent_message(messages: list[dict]) -> list[dict]:
        """
        Remove messages after the last agent message.
        """
        last_agent_message_index = None
        for i in reversed(range(len(messages))):
            if messages[i]["role"] == "assistant":
                last_agent_message_index = i
                break

        if last_agent_message_index is None:
            raise ValueError("No agent message found in the messages")
        return messages[: last_agent_message_index + 1]

    if isinstance(dataset_path, str):
        dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path {dataset_path} does not exist")
    dataset = SFTDataset.json_load(dataset_path)
    if split not in dataset:
        raise ValueError(f"Split {split} not found in dataset")
    datapoints = dataset[split]
    datapoints = [datapoint.model_dump() for datapoint in datapoints]
    num_discarded = 0
    openai_datapoints = []
    for datapoint in datapoints:
        if success_only and datapoint["reward"] != 1:
            num_discarded += 1
            continue
        openai_datapoint = OpenAICompletionDataPoint(
            messages=remove_messages_after_last_agent_message(datapoint["messages"]),
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
    do_save = True
    if save_path.exists():
        console.print(f"Dataset already exists at {save_path}")
        user_input = input("Overwrite? (y/n): ")
        if user_input.lower().strip() != "y":
            console.print("Exiting...")
            do_save = False
    if do_save:
        console.print(f"Saving dataset to {save_path}")
        with open(save_path, "w") as f:
            for datapoint in openai_datapoints:
                f.write(datapoint.model_dump_json() + "\n")
    return openai_datapoints


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
                        "id": tc.id,
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                        "type": "function",
                    }
                    for tc in message.tool_calls
                ]
            msg = {"role": "assistant"}
            if message.content:
                msg["content"] = message.content
            if tool_calls:
                msg["tool_calls"] = tool_calls
            openai_train_messages.append(msg)
        elif isinstance(message, ToolMessage):
            openai_train_messages.append(
                {
                    "tool_call_id": message.id,
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

    # Create mutually exclusive group for result paths
    result_group = make_parser.add_mutually_exclusive_group(required=True)
    result_group.add_argument(
        "--result-paths",
        type=str,
        nargs="+",
        help="Paths to result JSON files.",
    )
    result_group.add_argument(
        "--result-dir",
        type=str,
        help="Directory containing result JSON files.",
    )
    make_parser.add_argument(
        "--save-dir",
        type=str,
        required=True,
        help="Path to save the dataset.",
    )
    make_parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name of the dataset.",
    )
    make_parser.add_argument(
        "--success-only",
        action="store_true",
        help="Only include successful simulations.",
    )

    # Subparser for to_openai_sft_dataset
    convert_parser = subparsers.add_parser(
        "to-openai", help="Convert existing SFT dataset to OpenAI format"
    )
    convert_parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to the dataset.",
    )
    convert_parser.add_argument(
        "--save-dir",
        type=str,
        required=True,
        help="Path to save the dataset.",
    )
    convert_parser.add_argument(
        "--split",
        type=str,
        nargs="+",
        default=["train"],
        help="Splits to convert.",
    )
    convert_parser.add_argument(
        "--success-only",
        action="store_true",
        help="Only include successful simulations.",
    )

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    if args.command == "make":
        # Determine which argument was provided
        if args.result_paths:
            result_paths_or_dir = args.result_paths
        else:  # args.result_dir
            result_paths_or_dir = args.result_dir

        make_sft_dataset(
            result_paths_or_dir=result_paths_or_dir,
            save_dir=args.save_dir,
            name=args.name,
            success_only=args.success_only,
        )
    elif args.command == "to-openai":
        for split in args.split:
            console.print(f"Processing split: {split}")
            to_openai_sft_dataset(
                args.dataset_path, args.save_dir, split, args.success_only
            )


if __name__ == "__main__":
    main()
