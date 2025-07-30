from pathlib import Path
from typing import Callable, Optional

from datasets import Dataset, DatasetDict

from tau2.agent.base import is_valid_agent_history_message
from tau2.agent.llm_agent import LLMAgent
from tau2.data_model.message import Message, SystemMessage
from tau2.data_model.simulation import Results, SimulationRun
from tau2.data_model.tasks import Task
from tau2.environment.environment import Environment
from tau2.environment.tool import Tool
from tau2.registry import registry
from tau2.utils.llm_utils import to_litellm_messages
from tau2.utils.pydantic_utils import BaseModelNoExtra
from tau2.utils.utils import DATA_DIR

res_path = (
    DATA_DIR
    / "exp"
    / "qwen2.5-7b-retries"
    / "ollama_chat"
    / "qwen2.5:7b_telecom_no-user_gpt-4.1-2025-04-14_1trials.json"
)

save_path = DATA_DIR / "exp" / "data" / "test_sft_dataset"


def make_sft_dataset(
    result_path: str | Path, save_path: Optional[str | Path] = None
) -> DatasetDict:
    """
    Make a SFT dataset from a results file.
    """
    print(f"Loading results from {result_path}")
    results = Results.load(result_path)
    domain_name = results.info.environment_info.domain_name
    print(f"Domain: {domain_name}")
    agent_implementation = results.info.agent_info.implementation
    print(f"Agent implementation: {agent_implementation}")
    if "solo" in agent_implementation:
        solo_mode = True
    else:
        solo_mode = False
    print(f"Solo mode: {solo_mode}")
    print(f"Loading environment: Domain name: {domain_name}, Solo mode: {solo_mode}")
    environment: Environment = registry.get_env_constructor(domain_name)(
        solo_mode=solo_mode
    )
    tools: list[Tool] = environment.get_tools()
    if solo_mode:
        tools += environment.get_user_tools()
    print(f"Tools ({len(tools)}):\n{tools}"[:1000])
    openai_tools = [tool.openai_schema for tool in tools] if tools else None

    agent_constructor = registry.get_agent_constructor(agent_implementation)
    print(f"Agent constructor: {agent_constructor}")

    print(f"Building agent system prompt")

    def get_system_prompt(
        task,
    ):  # FIXME: This is not generic. Only works for solo mode.
        agent: LLMAgent = agent_constructor(
            tools=tools, domain_policy=environment.get_policy(), task=task
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
    for simulation in results.simulations:
        task = tasks[simulation.task_id]
        datapoint = make_sft_from_simulation(
            task, get_system_prompt, simulation, openai_tools
        )
        sft_dataset[get_split(task.id)].append(datapoint.model_dump())
    print(
        f"Number trajectories: {len(sft_dataset['train'])} (train) and {len(sft_dataset['test'])} (test)"
    )
    sft_dataset = DatasetDict(
        {
            "train": Dataset.from_list(sft_dataset["train"]),
            "test": Dataset.from_list(sft_dataset["test"]),
        }
    )
    if save_path is not None:
        print(f"Saving dataset to {save_path}")
        sft_dataset.save_to_disk(save_path)
    return sft_dataset


class SFTDataPoint(BaseModelNoExtra):
    messages: list[Message]
    tools: dict
    task_id: str
    simulation_id: str
    reward: float


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
    messages = to_litellm_messages(
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


def prepare_messages(messages: list[Message]) -> list[Message]:
    """
    Prepare the messages for the dataset.
    Filters out messages that are should not be part of the agent trajectory.
    """
    return [msg for msg in messages if is_valid_agent_history_message(msg)]
